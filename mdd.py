'''
A Multidimensional Data class (MDD) that offers name based storing and accession of data.
A flexible and generic data structure has been described below, and can be changed if required.
It provides complete name-based access to genes and subjects, independent of the order in which they are added/accessed.
(Note that because the structure is flexible, it might take a long time to populate very large datasets.)
'''

import numpy as np
from pprint import pprint
import pickle

class multidimensional_data:
    def __init__(self, info=[], name='data'): #Say info = [('samples',['subject_1', 'subject_2']),('genes',['gene_1','gene_2'])] >>> It's a good idea to give 'name' as the path to the data directory
        self.idname = name
        self.dims = len(info) #number of dimensions of object, say 2
        self.name = tuple([info[i][0] for i in range(len(info))]) #names of dimensions, say ('samples', 'genes')
        self.lens = [len(info[i][1]) for i in range(len(info))] #length of dimensions, say (10, 100)
        self.data = np.zeros(self.lens) #most important; the data table
        self.meta = [[{'name':info[i][1][j]} for j in range(len(info[i][1]))] for i in range(len(info))] #metadata for all variables, with compulsory metadata of 'name'
        self.forw_dict = [dict([(info[i][1][j], j) for j in range(len(info[i][1]))]) for i in range(len(info))]
        self.back_dict = [dict([(j, info[i][1][j]) for j in range(len(info[i][1]))]) for i in range(len(info))]
        self.smal_dict = dict([(info[i][0], i) for i in range(len(info))])
    
    def expand_dim_length(self, new_name, dim):
        if new_name not in self.forw_dict[dim]:                
            self.forw_dict[dim][new_name] = self.lens[dim]
            self.back_dict[dim][self.lens[dim]] = new_name
            self.lens[dim] += 1
            addindexs = self.lens[:]
            addindexs[dim] = 1
            self.data = np.concatenate((self.data, np.zeros(tuple(addindexs))), axis=dim)
            self.meta[dim].append({'name':new_name})

    def insert(self, nameidx, value): #Say nameidx = ('subject1', 'gene_1')
        for dim in range(self.dims): self.expand_dim_length(nameidx[dim], dim)
        numidx = tuple([self.forw_dict[dim][nameidx[dim]] for dim in range(self.dims)])
        self.data[numidx] = value

    def delete(self, dimname, dimval, metafield=''):
        dimnum = self.smal_dict[dimname]
        if metafield =='': #in case you want to delete an entire dimval
            if dimval in self.forw_dict[dimnum] and dimval not in ['name','platform']:
                mapping = self.forw_dict[dimnum].pop(dimval)
                self.data = np.delete(self.data, mapping, axis=dimnum)
                self.meta[dimnum] = self.meta[dimnum][:mapping] + self.meta[dimnum][mapping+1:]
                for key, value in self.forw_dict[dimnum].items():
                    if value>mapping:
                        self.forw_dict[dimnum][key] -= 1
                        self.back_dict[dimnum][value-1] = key
                self.lens[dimnum] -= 1
                self.back_dict[dimnum].pop(self.lens[dimnum])
        else: #in case you want to delete only a certain metadata of a particular dimval
            self.meta[dimnum][self.forw_dict[dimnum][dimval]].pop(metafield)

    def access(self, nameidx):
        try:
            numidx = tuple([self.forw_dict[dim][nameidx[dim]] for dim in range(self.dims)])
            return self.data[numidx]
        except: return None

    def metadata(self, dimname, dimval, metafield, metaval): #Say you wish to insert metafield 'infection' for dimname 'samples' and dimval 'subject_1' with metaval 'bacterial'
        dimnum = self.smal_dict[dimname]
        self.expand_dim_length(dimval, dimnum)
        self.meta[dimnum][self.forw_dict[dimnum][dimval]][metafield] = metaval

    def outvec(self, dimname, metafield=''):
        dimnum = self.smal_dict[dimname]
        if metafield == '': #in case you want a vector of only the dimension labels
            dimnum = self.smal_dict[dimname]
            vec = np.array(self.back_dict[dimnum].values())
        else: #generate a vector of some metadata
            vec = np.array([None for dimval in range(self.lens[dimnum])])
            for dimval in range(self.lens[dimnum]):
            	try: vec[dimval] = self.meta[dimnum][dimval][metafield]
            	except: print('metafield does not exist for', dimval)
        return vec.tolist()

    def displaydict(self, dimname):
        dimnum = self.smal_dict[dimname]
        pprint(self.back_dict[dimnum])

    def displaymeta(self, dimname, dimval):
        dimnum = self.smal_dict[dimname]
        pprint(self.meta[dimnum][self.forw_dict[dimnum][dimval]])

    def display(self):
        print('Name:', self.idname)
        print('Dimensions:', self.name)
        print('Length of dimensions:', self.lens)
        print('Data:')
        print(self.data)
        unique_metas = []
        for metainfo in self.meta:
        	unique_meta = dict()
        	for dim in metainfo:
        		for key in dim.keys():
        			if key != 'name':
        				if key in unique_meta:
        					if dim[key] in unique_meta[key]: unique_meta[key][dim[key]] += 1
	        				else: unique_meta[key][dim[key]] = 1
        				else: unique_meta[key] = {dim[key]:1}
        	unique_metas.append(unique_meta)
        print('Metadata tags:')
        pprint(unique_metas)

    def dump(self, name=''): #save the data instance as a Python object in a pickle dump
        name = self.idname+'/'+name+'_dump.pickle'
        pickle.dump(self, open(name, 'wb'))

    def writecsv(self, name=''): #save the data instance to a structured CSV file
        stream = [','.join(self.name)]
        for dim in self.name:
            stream.append(','.join(self.outvec(dim)))
        stack = [-1]
        while len(stack)!=0 :
            if len(stack) == self.dims:
                stack.pop()
                stream.append(','.join(['%f' %i for i in self.data[tuple(stack)]]))
            if stack[-1] < self.lens[len(stack)-1]-1:
                stack[-1] += 1
                stack.append(-1)
            else:
                stack.pop()
        for dimnum in range(self.dims):
            for dimvar in range(self.lens[dimnum]):
                stream.append(','.join([key+','+value for key, value in self.meta[dimnum][dimvar].items()]))
        name = self.idname+'/'+name+'_dump.csv'
        with open(name, 'w') as fd: fd.write('\n'.join(stream))

    def readcsv(self, csvfile, name='data'): #load the data instance from a structured CSV file
        self.idname = name
        with open(csvfile) as fd:
            self.name = tuple(fd.readline().strip().split(','))
            self.dims = len(self.name)
            i = 0
            dimvars = []
            while i<self.dims:
                dimvars.append(fd.readline().strip().split(','))
                i += 1
            self.lens = [len(dimvar) for dimvar in dimvars]
            self.meta = [[{} for j in range(len(dimvars[i]))] for i in range(len(dimvars))]
            self.forw_dict = [dict([(dimvars[i][j], j) for j in range(len(dimvars[i]))]) for i in range(len(dimvars))]
            self.back_dict = [dict([(j, dimvars[i][j]) for j in range(len(dimvars[i]))]) for i in range(len(dimvars))]
            self.smal_dict = dict([(self.name[i], i) for i in range(len(dimvars))])
            self.data = np.zeros(self.lens)
            stack = [-1]
            while len(stack)!=0 :
                if len(stack) == self.dims:
                    stack.pop()
                    self.data[tuple(stack)] = np.fromstring(fd.readline().strip(), dtype=float, sep=',')
                if stack[-1] < self.lens[len(stack)-1]-1:
                    stack[-1] += 1
                    stack.append(-1)
                else:
                    stack.pop()
            for dimnum in range(self.dims):
                for dimvar in range(self.lens[dimnum]):
                    metas = fd.readline().strip().split(',')
                    i = 0
                    while i < len(metas):
                        self.meta[dimnum][dimvar][metas[i]] = metas[i+1]
                        i += 2