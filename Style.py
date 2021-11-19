
def getBlackStyle():
    with open("./QSS/BlackStyle.qss") as file:
        string = file.readlines()
        string = ''.join(string)
    return string


def getDeepBlackStyle():
    with open("./QSS/DeepBlackStyle.qss") as file:
        string = file.readlines()
        string = ''.join(string)
    return string


def getMetroUIStyle():
    with open("./QSS/MetroUI.qss") as file:
        string = file.readlines()
        string = ''.join(string)
    return string
