'''
Parser for MINiML (XML) datasets, a common file format in which any GEO dataset can be downloaded.
MINiML (MIAME Notation in Markup Language, pronounced "minimal") is a data exchange format optimized for microarray gene expression data.

Some helpful instructions -->
1. Download the MINiML file and unzip it to an appropriate directory.
2. Go through the .xml schema file and find out your tags of interest, other than the usual data matrix of "samples X genes"
3. Change 'xmlfile' and 'metadata' variables appropriately (see end of file).
4. Run this Python script
	i. Open python in appropriate directory.
	ii. Import script by command >>> from parse_miniml import *
	iii. Use parse_xml() to generate a new parse, or load_miniml() to load an old parsed-dataset
	iv. Explore data in current Python session, or save using dump_all() function for later
'''

from mdd import * #we have defined a class in mdd.py
from xml.etree import ElementTree
import pickle

Namespace = '{http://www.ncbi.nlm.nih.gov/geo/info/MINiML}' #See xmlns in line 4 of XML file. You may not change this; universal for MINiML

def dump_all(parsed_xml, xmlfile):
	dumpfile = xmlfile.split('.')
	dumpfile = dumpfile[:-1]
	dumpfile = '.'.join(dumpfile)+'_miniml_dump.pickle'
	pickle.dump(parsed_xml, open(dumpfile, 'wb'))

def load_miniml(filename):
	return pickle.load(open(filename))

def parse_xml(xmlfile, metadata=[]):
    data_path = xmlfile.split('/')
    data_path = data_path[:-1]
    data_path = '/'.join(data_path)+'/'
    print('generating parse tree...')
    with open(xmlfile) as fd:
        tree = ElementTree.parse(fd)
    print('populating data table...')
    platform_data = dict()
    sample_flag = False
    sample_info = {'platform':'','name':''}
    otherstats = dict()
    for otherstat in metadata:
        if Namespace+otherstat[1] in otherstats:
            otherstats[Namespace+otherstat[1]].append(otherstat)
        else:
            otherstats[Namespace+otherstat[1]] = [otherstat]
    for node in tree.iter():
        if node.tag == Namespace+'Platform': #platform ID
            platform_data[node.attrib['iid']] = multidimensional_data([('samples',[]),('genes',[])], data_path)
            print('added platform', node.attrib['iid'])
        elif node.tag == Namespace+'Sample': #sample ID
            sample_flag = True
            sample_info['name'] = node.attrib['iid']
            print('for sample', node.attrib['iid'])
        elif node.tag == Namespace+'Platform-Ref' and sample_flag: #platform to which the sample belongs
            sample_info['platform'] = node.attrib['ref']
        elif node.tag in otherstats and sample_flag: #record other metadata of this sample
            for i in range(len(otherstats[node.tag])):
                if otherstats[node.tag][i][0] == 0:
                    sample_info[otherstats[node.tag][i][-1]] = node.text.strip()
                elif otherstats[node.tag][i][0] == 1:
                    sample_info[otherstats[node.tag][i][-1]] = node.attrib[otherstats[node.tag][i][2]]
                elif otherstats[node.tag][i][0] == 2:
                    sample_info[otherstats[node.tag][i][-2]] = node.attrib[otherstats[node.tag][i][2]]
                    sample_info[otherstats[node.tag][i][-1]] = node.text.strip()
                elif otherstats[node.tag][i][0] == 3:
                    if node.attrib[otherstats[node.tag][i][2]] == otherstats[node.tag][i][3]:
                        sample_info[otherstats[node.tag][i][-1]] = node.text.strip()
        elif node.tag == Namespace+'External-Data' and sample_flag: #record data table for this sample
            sample_flag = False
            data_file = node.text.strip()
            with open(data_path+'/'+data_file) as tablefile:
                for row in tablefile: #save all data
                    dat = row.split()
                    platform_data[sample_info['platform']].insert((sample_info['name'], dat[0]), float(dat[1])) #You might want to modify this depending on the data schema. Currently, assumes a table with col1 as gene_name and col2 as value.
            for key in sample_info.keys(): #save all metadata
                if key == 'name':
                    continue
                platform_data[sample_info['platform']].metadata('samples', sample_info['name'], key, sample_info[key])
            sample_info = {'platform':'','name':''}
    return platform_data
'''
Example variable below, with tsalik et al. (2016) and additional referenced datasets, thuong et al. (2008) dataset

xmlfile = {'tsalik':'../data/tsalik_2016/GSE63990_family.xml/GSE63990_family.xml', 'ramilo':'../data/tsalik_2016/GSE6269_family.xml/GSE6269_family.xml', 'herberg':'../data/tsalik_2016/GSE42026_family.xml/GSE42026_family.xml' 
		, 'hu':'../data/tsalik_2016/GSE40396_family.xml/GSE40396_family.xml',  'parnell':'../data/tsalik_2016/GSE20346_family.xml/GSE20346_family.xml', 'suarez': '../data/tsalik_2016/GSE60244_family.xml/GSE60244_family.xml'
		, 'bloom':'../data/tsalik_2016/GSE42834_family.xml/GSE42834_family.xml', 'thuong':'../data/thuong_2008/GSE11199_family.xml/GSE11199_family.xml'} #name of XML file
metadata = {'tsalik':[[0, 'Type'], [0, 'Organism'], [3, 'Characteristics', 'tag', 'infection_status']], 
			'ramilo':[[0, 'Type'], [0, 'Organism'], [3, 'Characteristics', 'tag', 'Illness'], [3, 'Characteristics', 'tag', 'Treatment'], [3, 'Characteristics', 'tag', 'Pathogen']],
			'herberg':[[0, 'Type'], [0, 'Organism'], [3, 'Characteristics', 'tag', 'infecting pathogen']],
			'hu':[[0, 'Type'], [0, 'Organism'], [3, 'Characteristics', 'tag', 'fever'],  [3, 'Characteristics', 'tag', 'pathogen']],
			'parnell':[[0, 'Type'], [0, 'Organism'], [0, 'Title']],
			'suarez':[[0, 'Type'], [0, 'Organism'], [3, 'Characteristics', 'tag', 'condition']],
			'bloom':[[0, 'Type'], [0, 'Organism'], [3, 'Characteristics', 'tag', 'disease state'], [3, 'Characteristics', 'tag', 'treatment']],
			'thuong':[[0, 'Type'], [0, 'Organism'], [0, 'Title']]}
minimlfile = {'tsalik':'../data/tsalik_2016/GSE63990_family.xml/GSE63990_family_miniml_dump.pickle', 'ramilo':'../data/tsalik_2016/GSE6269_family.xml/GSE6269_family_miniml_dump.pickle', 'herberg':'../data/tsalik_2016/GSE42026_family.xml/GSE42026_family_miniml_dump.pickle' 
		, 'hu':'../data/tsalik_2016/GSE40396_family.xml/GSE40396_family_miniml_dump.pickle',  'parnell':'../data/tsalik_2016/GSE20346_family.xml/GSE20346_family_miniml_dump.pickle', 'suarez': '../data/tsalik_2016/GSE60244_family.xml/GSE60244_family_miniml_dump.pickle'
		, 'bloom':'../data/tsalik_2016/GSE42834_family.xml/GSE42834_family_miniml_dump.pickle', 'thuong':'../data/thuong_2008/GSE11199_family.xml/GSE11199_family_miniml_dump.pickle'}

metadata is of the form -->
[[XML_read_code, XML_tag, XML_tag_att_field, XML_tag_att_val, XML_save_with_label_1, XML_save_with_label_2], ...]
where XML_read_codes can take values (entities in curly braces '{}' are optional):
	0 for simple save of (only) contents >>> [0, XML_tag{, XML_save_with_label_1}]
	1 for simple save of (only) tag-value >>> [1, XML_tag, XML_tag_att_field{, XML_save_with_label_1}]
	2 for simple save of (both) tag-value and contents >>> [2, XML_tag, XML_tag_att_field{, XML_save_with_label_1{, XML_save_with_label_2}}]
	3 for conditional save of (only) contents, if the tag value checks out (i.e. if XML_tag_att_field == XML_tag_att_val) >>> [3, XML_tag, XML_tag_att_field, XML_tag_att_val{, XML_save_with_label_1}]

The data schema is of the form -->
data = {<platform_name>:<platform_data_instance>}

Some helpful commands -->
1. To run the parser, use the parse_xml() function >>> parse_xml(<xml_file>,<metadata_instructions>) >>> Example: data = parse_xml(xmlfile['tsalik'], metadata['tsalik'])
2. To save a platform instance as a platform-CSV, use the multidimensional_data class function writecsv() to save it to directory containing the .xml file as >>> data['<platform_name>'].writecsv(<platform_name>) >>> Example: data['GPL571'].writecsv('GPL571')
3. To load  a platform instance from a platform-CSV, generate an empty multidimensional_data class instance, followed by the readcsv() function call to populate the empty variable >>> <data_new> = multidimensional_data(); <data_new>.readcsv(<platform_csv>)
4. To save a platform instance as a platform-pickle-dump, use the multidimensional_data class function dump() to save it to directory containing the .xml file as >>> data['<platform_name>'].dump(<platform_name>) >>> Example: data['GPL571'].dump('GPL571')
5. To save all instances as a pickle-dump, use the dump_all() function >>> dump_all(<data_var>,<xml_file>) >>> Example: dump_all(data, xmlfile['tsalik']) >>> Saves in same directory, with '_miniml_dump.pickle' suffix
6. To load any pickle-dump, use the load_miniml() function as >>> data = load_miniml(<pickle_file>) >>> Example: data = load_miniml(minimlfile['tsalik'])
7. To access data table, access the multidimensional_data class property data >>> data[<platform_name>].data >>> Since it's a numpy array, it can be used straight away for any ML algorithms.
8. To generate row vectors of appropriate metadata, which say can then be used as output labels in an ML algorithm, use the multidimensional_data class function outvec() >>> data['<platform_name>'].outvec(<'samples'/'genes'>,<sample_metaproperty/gene_metaproperty>) >>> Example: data['GPL571'].outvec('samples','infection')
9. To display metadata, use the multidimensional_data class function displaymeta() >>> data[<platform_name>].displaymeta(<'samples'/'genes'>,<sample_name/gene_name>) >>> Example: data['GPL571'].displaymeta('samples','GSM1561860')
10. To display the mapping of numeral and name variables, use the multidimensional_data class function displaydict() >>> data[<platform_name>].displaydict(<'samples'/'genes'>) >>> Example: data['GPL571'].access(('GSM1561860','211034_s_at'))
11. To display aggregate data properties, use the display function. Example: data[<platform_name>].display()
12. To access a particular data cell, use the multidimensional_data class function access() >>> data[<platform_name>].access((<sample_name>, <gene_name>)) >>> Example: data['GPL571'].displaymeta('samples','GSM1561860')
13. To delete a particular dimension variable's particular metadata, or an entire dimension variable, use the multidimensional_data class function delete() >>> data[<platform_name>].delete(<'samples'/'genes'>,<sample_name/gene_name>) >>> Example: data['GPL571'].delete('samples','GSM1561860')
14. Feel free to use other multidimensional_data class functions, for smaller manipulations like inserting/modifying metadata of subjects/genes, etc.
'''
xmlfile = {'hsecs':'../../../Downloads/GSE52158_family.xml/GSE52158_family.xml'} #name of XML file
metadata = {'hsecs':[[0, 'Title']]}
minimlfile = {'hsecs':'../../../Downloads/GSE52158_family.xml/GSE52158_family_miniml_dump.pickle'}
data = parse_xml(xmlfile['hsecs'], metadata['hsecs'])
data['GPL570'].writecsv('GPL570')