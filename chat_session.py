class ChatSession:
    def __init__(self, provider):
        self.provider = provider
        self.history = []

    def send(self, message):
        response = self.provider.send_message(message, self.history)
        self.history.append({"user": message, "assistant": response})
        return response
