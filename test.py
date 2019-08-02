class test(object):
    def __init__(self):
        self.counter = 0
        for i in range(5):
            self.counter += 1
    def function(self):
        print(self.counter)

t = test()
t.function()