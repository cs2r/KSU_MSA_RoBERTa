from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer('vocab/vocab.json', 'vocab/merges.txt')

tokenizer.enable_truncation(max_length=512)

token_text = tokenizer.encode("<number> hdG jRGd JYGWj GdehGO GdGajhfjI, dG SjeG aj Tcd GdgjQhjf, jJSHH aj eYXe WdH YdGL JYGWj GdeNOQGJ aj eYXe HdOGf hSW hTQb hLfhH TQb GSjG.")
print(token_text)

print("done")