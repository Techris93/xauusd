self.SIGNAL_PUSH_PAYLOAD_VERSION = "authoritative-signal-v2";
self.lastSignalPush = null;

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

  const tag = payload.tag || "xauusd-important-signal";
  if (
    tag === "xauusd-important-signal" &&
    payload.version !== self.SIGNAL_PUSH_PAYLOAD_VERSION
  ) {
    console.warn("Ignoring stale signal push payload.");
    return;
  }
  if (tag === "xauusd-important-signal") {
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
  if (tag === "xauusd-important-signal") {
    const dedupeKey = payload.dedupeKey || `${title}|${body}`;
    const lastPush = self.lastSignalPush;
    const duplicateWindowSeconds = Number.isFinite(Number(payload.maxAgeSeconds))
      ? Number(payload.maxAgeSeconds)
      : 300;
    if (
      lastPush &&
      lastPush.key === dedupeKey &&
      Date.now() - lastPush.time < duplicateWindowSeconds * 1000
    ) {
      console.warn("Ignoring duplicate signal push payload.");
      return;
    }
    self.lastSignalPush = {
      key: dedupeKey,
      time: Date.now(),
    };
  }

  const options = {
    body: body,
    tag: tag,
    renotify: true,
    requireInteraction: Boolean(payload.requireInteraction),
    data: {
      url: payload.url || "/",
    },
  };

  event.waitUntil(
    (async () => {
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
