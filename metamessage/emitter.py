"""An owned group of registrations using the existing injection mechanism."""

from .models import MetaMessageLifecycle


class MetaMessageEmitter:
    """A module can release its registrations without erasing permanent history."""

    def __init__(self, manager):
        self.manager = manager
        self._messages = {}
        self._subscriptions = []
        self._closed = False

    def _track(self, message):
        if self._closed:
            raise RuntimeError("Message emitter is closed")
        self._messages[message.message_id] = message.lifecycle
        return message

    def emit(self, message):
        return self.manager.emit(self._track(message))

    def register(self, message):
        return self.manager.register(self._track(message))

    def subscribe(self, event_name, factory):
        def tracked(context):
            result = factory(context)
            if result is None:
                return None
            if isinstance(result, (list, tuple)):
                return [self._track(item) for item in result]
            return self._track(result)
        self.manager.subscribe(event_name, tracked)
        self._subscriptions.append((event_name, tracked))

    def close(self):
        for event, factory in self._subscriptions:
            self.manager.unsubscribe(event, factory)
        for message_id, lifecycle in self._messages.items():
            self.manager.unregister(message_id, remove_injected=lifecycle != MetaMessageLifecycle.PERMANENT)
        self._subscriptions.clear()
        self._messages.clear()
        self._closed = True
