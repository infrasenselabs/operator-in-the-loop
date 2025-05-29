import infradb2
db = infradb2.InfraDB2('infradb2')
r = db.frame('query database')
print(r)