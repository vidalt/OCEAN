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
max_depths = range(3,9)

write = open("scalabilityDepth_FT.csv","w")

write.write('dataset,sample,maxDepth,nbTrees,alg,cfe_distance,cfe_found,cfe_plausible,cfe_time,\n')

for ds in datasets:
    for md in max_depths:
        print('FT',ds,md,100, 'open')
        folder = findFolderName(ds,'FT',md,100)
        if folder:
            filename = "./_experiments/" + folder + "/minimum_distances.txt"
            if os.path.isfile(filename):
                read = open(filename)
                stringFile = read.read().replace("\'", "\"")
                results = eval(stringFile)
                print(filename)
                for sample in results:
                    write.write(ds+",")
                    write.write(sample+",")
                    write.write(str(md)+',')
                    write.write(str(100)+',')
                    write.write('FT,')
                    write.write(str(results[sample]['cfe_distance'])+',')
                    write.write(str(results[sample]['cfe_found'])+',')
                    write.write(str(results[sample]['cfe_plausible'])+',')
                    write.write(str(results[sample]['cfe_time'])+',')
                    write.write("\n")
            else:
                print(filename, 'not found')
        else:
            print('FT',ds,md,100, 'not found')


