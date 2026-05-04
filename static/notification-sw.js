self.SIGNAL_PUSH_PAYLOAD_VERSION = "authoritative-signal-v2";
self.lastSignalPush = null;
self.recentSignalEventIds = new Map();

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("push", (event) => {
  let payload = {};

  if (event.data) {
    try {
      payload = event.data.json();
    } catch (error) {
      payload = {
        body: event.data.text(),
      };
    }
  }

  const eventId =
    payload.notificationEventId ||
    payload.notification_event_id ||
    payload.dedupeKey ||
    null;
  const tag = payload.tag || eventId || "xauusd-important-signal";
  const isSignalPayload =
    payload.pushType === "signal" ||
    payload.version === self.SIGNAL_PUSH_PAYLOAD_VERSION ||
    tag === "xauusd-important-signal" ||
    String(tag).startsWith("xauusd-");
  if (isSignalPayload && payload.version !== self.SIGNAL_PUSH_PAYLOAD_VERSION) {
    console.warn("Ignoring stale signal push payload.");
    return;
  }
  if (isSignalPayload) {
    const createdAt = Date.parse(payload.createdAt || "");
    const maxAgeSeconds = Number.isFinite(Number(payload.maxAgeSeconds))
      ? Number(payload.maxAgeSeconds)
      : 300;
    if (
      !Number.isFinite(createdAt) ||
      Date.now() - createdAt > maxAgeSeconds * 1000
    ) {
      console.warn("Ignoring expired signal push payload.");
      return;
    }
  }

  const title = payload.title || "XAU/USD signal update";
  const body = payload.body || "Important signal conditions changed.";
  if (isSignalPayload) {
    const dedupeKey = eventId || `${title}|${body}`;
    const lastPush = self.lastSignalPush;
    const duplicateWindowSeconds = Number.isFinite(Number(payload.maxAgeSeconds))
      ? Number(payload.maxAgeSeconds)
      : 300;
    const now = Date.now();
    for (const [storedEventId, storedAt] of self.recentSignalEventIds.entries()) {
      if (now - storedAt > duplicateWindowSeconds * 1000) {
        self.recentSignalEventIds.delete(storedEventId);
      }
    }
    if (self.recentSignalEventIds.has(dedupeKey)) {
      console.warn("Ignoring duplicate signal event payload.", {
        notificationEventId: dedupeKey,
      });
      return;
    }
    if (
      lastPush &&
      lastPush.key === dedupeKey &&
      now - lastPush.time < duplicateWindowSeconds * 1000
    ) {
      console.warn("Ignoring duplicate signal push payload.");
      return;
    }
    self.recentSignalEventIds.set(dedupeKey, now);
    self.lastSignalPush = {
      key: dedupeKey,
      time: now,
    };
  }

  const options = {
    body: body,
    tag: tag,
    renotify: false,
    requireInteraction: Boolean(payload.requireInteraction),
    data: {
      url: payload.url || "/",
      notificationEventId: eventId,
    },
  };

  event.waitUntil(
    (async () => {
      if (isSignalPayload) {
        const clients = await self.clients.matchAll({
          type: "window",
          includeUncontrolled: true,
        });
        for (const client of clients) {
          client.postMessage({
            type: "xauusd.signalPush",
            createdAt: payload.createdAt || null,
            snapshotId: payload.signalSnapshotId || null,
            notificationEventId: eventId,
          });
        }
      }
      await self.registration.showNotification(title, options);
    })(),
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  const targetUrl =
    event.notification && event.notification.data && event.notification.data.url
      ? event.notification.data.url
      : "/";

  event.waitUntil(
    (async () => {
      const clients = await self.clients.matchAll({
        type: "window",
        includeUncontrolled: true,
      });

      for (const client of clients) {
        if ("focus" in client) {
          await client.focus();
          if (client.url !== targetUrl && "navigate" in client) {
            await client.navigate(targetUrl);
          }
          return;
        }
      }

      if (self.clients.openWindow) {
        await self.clients.openWindow(targetUrl);
      }
    })(),
  );
});
