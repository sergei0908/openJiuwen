import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

tokens = encoding.encode("hello world")
print(tokens)               # [15339, 1917]
print(len(tokens))          # 2
print(encoding.decode(tokens))  # hello world