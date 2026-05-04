self.SIGNAL_PUSH_PAYLOAD_VERSION = "authoritative-signal-v2";
self.SIGNAL_NOTIFICATION_EVENT_CACHE = "xauusd-signal-notification-events-v1";
self.SIGNAL_NOTIFICATION_EVENT_PREFIX = "__xauusd_notification_event__/";
self.lastSignalPush = null;
self.recentSignalEventIds = new Map();

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

function signalEventRequestUrl(eventId) {
  return new URL(
    `${self.SIGNAL_NOTIFICATION_EVENT_PREFIX}${encodeURIComponent(eventId)}`,
    self.registration.scope,
  ).toString();
}

async function pruneStoredSignalEventIds(cache, maxAgeSeconds, now) {
  const requests = await cache.keys();
  await Promise.all(
    requests.map(async (request) => {
      if (!request.url.includes(self.SIGNAL_NOTIFICATION_EVENT_PREFIX)) {
        return;
      }
      const response = await cache.match(request);
      const payload = response ? await response.json().catch(() => null) : null;
      const displayedAt = payload ? Number(payload.displayedAt) : NaN;
      if (!Number.isFinite(displayedAt) || now - displayedAt > maxAgeSeconds * 1000) {
        await cache.delete(request);
      }
    }),
  );
}

async function hasStoredSignalEventId(cache, eventId) {
  if (!eventId) {
    return false;
  }
  return Boolean(await cache.match(signalEventRequestUrl(eventId)));
}

async function rememberStoredSignalEventId(cache, eventId, payload, now) {
  if (!eventId) {
    return;
  }
  await cache.put(
    signalEventRequestUrl(eventId),
    new Response(
      JSON.stringify({
        notificationEventId: eventId,
        displayedAt: now,
        createdAt: payload.createdAt || null,
        snapshotId: payload.signalSnapshotId || null,
      }),
      {
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "max-age=86400",
        },
      },
    ),
  );
}

async function broadcastSignalPush(payload, eventId, status) {
  const clients = await self.clients.matchAll({
    type: "window",
    includeUncontrolled: true,
  });
  for (const client of clients) {
    client.postMessage({
      type: "xauusd.signalPush",
      createdAt: payload.createdAt || null,
      snapshotId: payload.signalSnapshotId || null,
      rawSignalSnapshotId: payload.rawSignalSnapshotId || null,
      notificationEventId: eventId,
      producer: "service_worker",
      osNotificationAttempted: status === "displayed",
      displayStatus: status,
    });
  }
}

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
  const title = payload.title || "XAU/USD signal update";
  const body = payload.body || "Important signal conditions changed.";

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
        if (payload.version !== self.SIGNAL_PUSH_PAYLOAD_VERSION) {
          console.warn("Ignoring stale signal push payload.", {
            notificationEventId: eventId,
            producer: "service_worker",
            osNotificationAttempted: false,
            skipReason: "stale_payload_version",
          });
          await broadcastSignalPush(payload, eventId, "skipped_stale");
          return;
        }

        const createdAt = Date.parse(payload.createdAt || "");
        const duplicateWindowSeconds = Number.isFinite(Number(payload.maxAgeSeconds))
          ? Number(payload.maxAgeSeconds)
          : 300;
        const now = Date.now();
        if (
          !Number.isFinite(createdAt) ||
          now - createdAt > duplicateWindowSeconds * 1000
        ) {
          console.warn("Ignoring expired signal push payload.", {
            notificationEventId: eventId,
            producer: "service_worker",
            osNotificationAttempted: false,
            skipReason: "expired_payload",
          });
          await broadcastSignalPush(payload, eventId, "skipped_expired");
          return;
        }

        const dedupeKey = eventId || `${title}|${body}`;
        const lastPush = self.lastSignalPush;
        let eventCache = null;
        try {
          eventCache = await caches.open(self.SIGNAL_NOTIFICATION_EVENT_CACHE);
          await pruneStoredSignalEventIds(eventCache, duplicateWindowSeconds, now);
        } catch (error) {
          console.warn("Persistent signal event dedupe unavailable.", {
            notificationEventId: dedupeKey,
            producer: "service_worker",
            osNotificationAttempted: false,
            skipReason: "persistent_dedupe_unavailable",
            error: error && error.message ? error.message : String(error),
          });
        }
        for (const [storedEventId, storedAt] of self.recentSignalEventIds.entries()) {
          if (now - storedAt > duplicateWindowSeconds * 1000) {
            self.recentSignalEventIds.delete(storedEventId);
          }
        }
        if (eventCache && (await hasStoredSignalEventId(eventCache, dedupeKey))) {
          console.warn("Ignoring duplicate signal event payload.", {
            notificationEventId: dedupeKey,
            producer: "service_worker",
            osNotificationAttempted: false,
            skipReason: "persistent_event_seen",
          });
          await broadcastSignalPush(payload, eventId, "skipped_duplicate");
          return;
        }
        if (self.recentSignalEventIds.has(dedupeKey)) {
          console.warn("Ignoring duplicate signal event payload.", {
            notificationEventId: dedupeKey,
            producer: "service_worker",
            osNotificationAttempted: false,
            skipReason: "memory_event_seen",
          });
          await broadcastSignalPush(payload, eventId, "skipped_duplicate");
          return;
        }
        if (
          lastPush &&
          lastPush.key === dedupeKey &&
          now - lastPush.time < duplicateWindowSeconds * 1000
        ) {
          console.warn("Ignoring duplicate signal push payload.", {
            notificationEventId: dedupeKey,
            producer: "service_worker",
            osNotificationAttempted: false,
            skipReason: "last_push_seen",
          });
          await broadcastSignalPush(payload, eventId, "skipped_duplicate");
          return;
        }
        self.recentSignalEventIds.set(dedupeKey, now);
        self.lastSignalPush = {
          key: dedupeKey,
          time: now,
        };
        if (eventCache) {
          await rememberStoredSignalEventId(eventCache, dedupeKey, payload, now);
        }
        await broadcastSignalPush(payload, eventId, "displayed");
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
