import csv

# Our Classes
classes = ["spam", "ham"]

def load_spam_data():

    # Data File
    file = "data/spam.csv"

    # Lists to store all word frequencies from messages
    # used to find top words for each class
    content_spam = {}
    content_ham = {}

    # Lists for each message and corresponding label
    messages = []
    y = []

    # open the file for processing as a CSV
    with open(file, 'r') as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            if i == 0:
                continue

            # split the string and remove all non alpha characters (or ')
            words = [''.join(c for c in word if c.isalpha() or c == "'") for word in row[1].lower().split()]

            # Add and count words for spam and ham classes
            content = content_spam if row[0] == "spam" else content_ham
            for w in words:
                if len(w) > 4:
                    if w in content:
                        content[w] += 1
                    else:
                        content[w] = 1

            # Append full messages
            messages.append(" ".join(words))
            y.append(classes.index(row[0]))

    # sort each each word based on value count
    sorted_X_spam = sorted(content_spam, key=content_spam.get, reverse=True)
    sorted_X_ham = sorted(content_ham, key=content_ham.get, reverse=True)

    # populate the bag-of-words with top 50 words from each class (and remove duplicates)
    bow = []
    for i in range(50):
        if sorted_X_spam[i] not in bow:
            bow.append(sorted_X_spam[i])
        if sorted_X_ham[i] not in bow:
            bow.append(sorted_X_ham[i])


    return bow, messages, y