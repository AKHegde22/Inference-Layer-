REVIEW_GRAMMAR = """
root   ::= "{" ws "\"product\":" ws string "," ws "\"rating\":" ws rating "," ws "\"recommended\":" ws bool "," ws "\"summary\":" ws string ws "}"
string ::= "\"" [a-zA-Z0-9 .,!'-]+ "\""
rating ::= [1-5]
bool   ::= "true" | "false"
ws     ::= [ \t\n]*
"""

print(REVIEW_GRAMMAR)