
class Tweet:

    def __init__(self, *args):
        self.title = ""
        self.body = ""
        self.text = ""
        self.domain = ""
        self.date = ""
        self.interactions = 0
        self.truthfulness = 0
        self.conversation_id = 0
        self.image = None

        if len(args) > 1:
            self.title = args[0]
            self.body = args[1]
            self.text = args[2]
            self.interactions = args[3]
            self.conversation_id = args[4]
            self.image = args[5]
            self.domain = args[6]
            self.date = args[7]

    def to_json(self):
        return self.__dict__
