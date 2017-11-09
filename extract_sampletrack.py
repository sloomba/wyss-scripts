import json
from pprint import pprint

class st_templates:
	def __init__(self, jsonname):
		self.name = '.'.join(jsonname.split('.')[:-1])
		with open(jsonname) as jsonfile:
			data = json.load(jsonfile)
		self.data = dict([(datum['_id'], datum) for datum in data])
		nodes = set(self.data.keys())
		graphs = dict() #each graph corresponds to one rooted graph
		while len(nodes)!=0:
			root = nodes.pop() #take any unvisited node as an assumed root
			stack = [(root, [], 1)] #depth-first exploration using stack
			while len(stack)>0:
				node_tup = stack[-1]
				if node_tup[0] not in node_tup[1]: #if node doesn't cycle back to itself, just in case graph is not DAG
					if node_tup[0] in graphs.keys(): #visited, thus sub-tree complete --> update graph connections
						stack.pop(-1)
						graphs[node_tup[0]] = (node_tup[2], dict([(child['_id'], graphs.pop(child['_id'])) for child in self.data[node_tup[0]]['components'].values()]))
					else: #not visited, thus explore sub-tree
						graphs[node_tup[0]] = (node_tup[2], dict()) #marks it as visited
						children = [(child['_id'], child['quantity']) for child in self.data[node_tup[0]]['components'].values()]
						for child in children:
							graphs.pop(child[0], None) #removes "wrong" previous iterations where children were assumed as roots of trees
							nodes.discard(child[0]) #remove children from list of "potential roots"
							stack.append((child[0], node_tup[1]+[node_tup[0]], child[1]))
				else: #if node cycles back to itself, terminate further exploration
					graphs[node_tup[0]] = (node_tup[2], dict()) #because it might have been 'unvisited' in line: graphs.pop(child[0], None)
					stack.pop(-1)
		self.graph = graphs

	def parse_dfs(self):
		out = []
		for root in self.graph.keys():
			stack = [(root, self.graph[root], 0)]
			while len(stack)>0:
				node = stack.pop(-1)
				out.append('\t'*node[2]+'ID: '+str(node[0])+', NAME: '+self.data[node[0]]['name']+', QTY: '+str(node[1][0])+'\n'+'\t'*node[2]+'PROP: '+', '.join(self.data[node[0]]['properties'].keys())+'\n'+'\t'*node[2]+'PROC: '+', '.join([proc['name'] for proc in self.data[node[0]]['processes']]))
				#out.append('\t'*node[2]+self.data[node[0]]['name']+' * '+str(node[1][0]))
				for child in node[1][1].keys():
					stack.append((child, node[1][1][child], node[2]+1))
		out = '\n\n'.join(out)
		with open(self.name+'_parse.txt','wb') as fd:
			fd.write(out)
		return out

	def enlist_attributes(self):
		props = set() #can go in "study" table?
		procs = set()
		for val in self.data.values():
			waste = [props.add(key) for key in val['properties'].keys()]
			waste = [procs.add(proc['name']) for proc in val['processes']]
		pprint(props)
		pprint(procs)

class st_parts:
	def __init__(self, jsonname):		
		self.name = '.'.join(jsonname.split('.')[:-1])
		with open(jsonname) as jsonfile:
			data = json.load(jsonfile)
		self.data = dict([(datum['_id'], datum) for datum in data])
		nodes = set(self.data.keys())
		graphs = dict() #each graph corresponds to one rooted graph
		while len(nodes)!=0:
			root = nodes.pop() #take any unvisited node as an assumed root
			stack = [(root, [])] #depth-first exploration using stack
			while len(stack)>0:
				node_tup = stack[-1]
				if node_tup[0] not in node_tup[1]: #if node doesn't cycle back to itself, just in case graph is not DAG
					if node_tup[0] in graphs.keys(): #visited, thus sub-tree complete --> update graph connections
						stack.pop(-1)
						try:
							graphs[node_tup[0]] = dict([(int(temp_id), [(child, graphs.pop(child)) for child in children['val']]) for (temp_id, children) in self.data[node_tup[0]]['data']['components'].iteritems()])
						except:
							print 'some children of part with ID', node_tup[0], 'not found'
					else: #not visited, thus explore sub-tree
						graphs[node_tup[0]] = dict() #marks it as visited
						try:
							children = sum([children['val'] for children in self.data[node_tup[0]]['data']['components'].values()], [])
							for child in children:
								graphs.pop(child, None) #removes "wrong" previous iterations where children were assumed as roots of trees
								nodes.discard(child) #remove children from list of "potential roots"
								stack.append((child, node_tup[1]+[node_tup[0]]))
						except:
							print 'part with ID', node_tup[0], 'and ancestors', node_tup[1], 'not found'
							stack.pop(-1)
				else: #if node cycles back to itself, terminate further exploration
					graphs[node_tup[0]] = dict() #because it might have been 'unvisited' in line: graphs.pop(child[0], None)
					stack.pop(-1)
		self.graph = graphs

	def enlist_attributes(self):
		procs = set()
		users = set()
		for val in self.data.values():
			waste = [procs.add(key) for key in val['data']['processes'].keys()]
			users.add((val['user']['_id'], val['user']['name']))
		pprint(procs)
		pprint(users)

x = st_templates('../data/sampletrack/sampletrack_templates.txt')
y = st_parts('../data/sampletrack/sampletrack_parts.txt')
#x.enlist_attributes()
#y.enlist_attributes()
x.parse_dfs()