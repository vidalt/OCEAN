import os

def findFolderName(ds,alg,md,ne):
    for root,dir,path in os.walk("./_experiments"):
        folders = root.split('/')
        if len(folders) > 2:
            folder = folders[2]
            if "_" + ds + "_" in folder:
                if '_FT_' in folder: 
                    if "_md" + str(md) + "_" in folder:
                        temp = folder + "_"
                        if "_ne" + str(ne) +"_" in temp:
                            return folder
    return False

datasets = ['compass','adult','credit']
nbTrees = [10,20,50,100,200,500]

write = open("scalabilityNbTrees_FT.csv","w")

write.write('dataset,sample,maxDepth,nbTrees,alg,cfe_distance,cfe_found,cfe_plausible,cfe_time,\n')

for ds in datasets:
    for ne in nbTrees:
        folder = findFolderName(ds,'FT',5,ne)
        print('FT',ds,5,ne)
        if folder:
            filename = "./_experiments/" + folder + "/minimum_distances.txt"
            print(filename)
            read = open(filename)
            stringFile = read.read().replace("\'", "\"")
            results = eval(stringFile)
            for sample in results:
                write.write(ds+",")
                write.write(sample+",")
                write.write(str(5)+',')
                write.write(str(ne)+',')
                write.write('FT,')
                write.write(str(results[sample]['cfe_distance'])+',')
                write.write(str(results[sample]['cfe_found'])+',')
                write.write(str(results[sample]['cfe_plausible'])+',')
                write.write(str(results[sample]['cfe_time'])+',')
                write.write("\n")


