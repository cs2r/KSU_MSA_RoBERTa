from transformers import pipeline
from pandas import read_excel


data = read_excel("ASMO_table.xlsx",
                  sheet_name="ASMO_table - Org - Copy").values.tolist()
dictIn = []
dictOut = []
for element in data:
    dictIn.append(element[0].split(",")[0])
    dictOut.append(element[0].split(",")[1])
dictIn = ''.join(dictIn)
dictOut = ''.join(dictOut)

fill_mask = pipeline("fill-mask", model="model0.1",
                     tokenizer="model0.1")

with open('../dataset/test.txt') as f:
    data = f.readlines()

while True:
    lineNdx = input("Please enter the number between 0 and " + str(len(data)) + ": ")
    if lineNdx == 'x':
        break
    test = data[int(lineNdx)]
    translation = test.maketrans(dictIn, dictOut)
    translated = test.translate(translation)
    strList = translated.split()
    translated = ''
    for ndx in range(len(strList)):
        translated = translated + str(ndx) + ":" + strList[ndx] + '   '
    print(translated)
    ndx = input("Please enter the number of the word that you want to MASK: ")
    if ndx == 'x':
        continue

    original = test.split()[int(ndx)]
    test = test.replace(original, "<mask>")
    results = fill_mask(test)
    text = []
    print("\n****************************************** RESULTS **************************************************\n")

    for result in results:
        sent = result["sequence"]
        translation = sent.maketrans(dictIn, dictOut)
        translated = sent.translate(translation)
        text.append(translated + " Score: " + str(result["score"]))

    for line in text:
        print(line)
