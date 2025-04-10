''' test2.py

Input:
    1. 'embeddings' csv file
    2. 'filtered_enum' csv file
    3. 'seen' pickle file
    4. Str

Function:
    1. Load file data.
    2. If previous files have been loaded, update variables to accept new words.
    2. Change associative probabilities between adjacent words based on new str.
    3. Save key files.

Output:
    1. 'embeddings' csv file
    2. 'filtered_enum' csv file
    3. 'seen' pickle file

TODO - Turn embedding matrix into sparse matrix.
TODO - Turn code into external functions where possible.
'''

import numpy as np
import pickle
import random
from scipy.sparse import csr_matrix

# import existing csv 'embedding.csv'
try:
    embedding = np.loadtxt("embedding.csv", delimiter=",")
except:
    pass

# If not exist
if "embedding" not in globals():
    ## input sentence
    sentence = "this is one sentence that will repeat itself at least one time"
    ## enumerate words in sentence
    enum  = list(enumerate(sentence.split()))
    ## Extract list of words that are not duplicated.
    seen = set()
    filteredEnum = [(index, word) for index, word in enum if word not in seen and not seen.add(word)]
    ## create square array of zeros with dim size(index)
    embedding = np.zeros([len(filteredEnum), len(filteredEnum)], float)
else:
    # Process new sentence and add new words to filteredEnum
    # Load CSV file
    data = np.loadtxt("filteredEnum.csv", delimiter=",", dtype=str)
    # Convert to a list of tuples
    filteredEnum = [(int(index), word) for index, word in data]
    filteredEnumLen = len(filteredEnum)
    # Load the set from the file
    with open("seen.pkl", "rb") as file:
        seen = pickle.load(file)
    newSentence = input("Enter your input here: ") #"this is another sentence and will repeat itself a few more times and will include a new word or more"
    newEnum = list(enumerate(newSentence.split(), start=len(filteredEnum)))
    for index, word in newEnum:
        if word not in seen:  # Check if the word is new
            filteredEnum.append((index, word))
            seen.add(word)  # Add to seen set
    filteredEnumLen2 = len(filteredEnum)
    ## save newEnum as enum
    enum = newEnum
    # Extend embedding matrix to include new words.
    ## Find how many new words there are to add to embedding.
    newWordNum = filteredEnumLen2 - filteredEnumLen
    ## for each row append a new array of np.zeros(len(newWords))
    ## New columns (must have the same number of rows)
    newColumns = np.zeros((embedding.shape[0], newWordNum))
    embedding = np.hstack((embedding, newColumns))
    newRows = np.zeros((newWordNum, len(filteredEnum)))
    embedding = np.vstack((embedding, newRows))
    ## for each new word, add a new row of np.zeros(len(filteredEnum))


## For each word, pair with following word.
for index, (item, nextItem) in enumerate(zip(enum, enum[1::])):
    breakout = 0
    print(f"Index: {index}, Current Word: {item[1]}, Next Word: {nextItem[1]}")
    ### Check the current word for its position in the embeddings
    for x in range(len(filteredEnum)):
        ### when it is found find the position of the second word.
        if item[1] == filteredEnum[x][1]:
            for y in range(len(filteredEnum)):
                if nextItem[1] == filteredEnum[y][1]:
                    ### For the word pair, adjust the probability in the embedding matrix.
                    embedding[x][y] = embedding[x][y] + 1
                    breakout = 1
                if breakout:
                    break
        if breakout:
            break

# save as csv 'embedding.csv'
np.savetxt("embedding.csv", embedding, delimiter=",", fmt="%s")

# Convert list of tuples to an array
filteredEnum_array = np.array(filteredEnum, dtype=object)  # Use dtype=object for mixed types

# Save to CSV
np.savetxt("filteredEnum.csv", filteredEnum_array, delimiter=",", fmt="%s")

# Save the set to a file
with open("seen.pkl", "wb") as file:
    pickle.dump(seen, file)


# Create a normalised embedding matrix when determining the next word.

##  normalise values across each row.
normEmbedding = np.zeros([len(embedding), len(embedding)])

## For each row.
for row in range(len(normEmbedding)):
    totalInRow = 0
    ### sum row values
    for col in range(len(normEmbedding)):
        totalInRow = totalInRow + embedding[row][col]
    ### for each element divide value by total sum
    for col in range(len(normEmbedding)):
        if totalInRow == 0:
            normEmbedding[row][col] = 0
            continue
        normEmbedding[row][col] = embedding[row][col]/totalInRow

numOfWords = 10
# Start with random word.
## Find index of initial word
wordIndex = filteredEnum.index(filteredEnum[int(random.random()*len(filteredEnum))])
for word in range(numOfWords):
    ## For row of embedding matrix of index, Use random number to choose the next word in line with the probabilities on that row.
    threshold = random.random()
    summer = 0
    for col in range(len(normEmbedding)):
        summer = summer + normEmbedding[wordIndex][col]
        ## Once selected, find the column index for that selection.
        if summer > threshold:
            ## With that column index, find the word in filtered_enum.
            nextWord = filteredEnum[col][1]
            ## Add word to word list.
            print(nextWord)
            wordIndex = col
            break
        else:
            continue
    
end = input("Press enter to end program")
