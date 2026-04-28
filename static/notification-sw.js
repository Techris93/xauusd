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

  const title = payload.title || "XAU/USD signal update";
  const options = {
    body: payload.body || "Important signal conditions changed.",
    tag: payload.tag || "xauusd-important-signal",
    renotify: true,
    requireInteraction: Boolean(payload.requireInteraction),
    data: {
      url: payload.url || "/",
    },
  };

  event.waitUntil(
    (async () => {
      const clients = await self.clients.matchAll({
        type: "window",
        includeUncontrolled: true,
      });
      const hasVisibleClient = clients.some(
        (client) => client.visibilityState === "visible" || client.focused,
      );

      if (hasVisibleClient) {
        return;
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
