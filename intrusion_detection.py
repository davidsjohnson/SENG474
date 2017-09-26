
import csv
import pydotplus as pydot

from sklearn import tree
from sklearn.externals.six import StringIO

"""
This program creates a decision tree based off of data from live hacking incidents.
Decision trees are used for intrusion detection much in the same way.

DATA TAKEN FROM: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

This program has some requirements that are NOT installed on the ELW's lab machines.
Here is an installation guide:

	1. Make sure you have python installed (I am using python 2.7).
	2. Make sure you have scikit-learn installed (Tutorial at http://calebshortt.com/2016/01/15/installing-scikit-learn-python-data-mining-library/)
	3. Install graphviz (http://www.graphviz.org/Download..php)
	4. (Windows) Set your env path to include where you installed graphviz (specifically the bin folder) (mine was C:\Program Files (x86)\Graphviz2.38\bin)
	5. Install the python-graghviz link: pip install graphviz
	6. Install pydot2: pip install pydot2
	7. Test by opening a comand line terminal and typing:
			python  		(opens the python interpreter and prompts with '>>>')
			import pydot 	(if nothing happens and a new prompt comes up, all is good.)


"""


class DataFormatting:

	class Mappings:
		"""
		Mappings are important. Scikit-learn ONLY deals with numerical data -- specifically integers.
		This means that we have to map each item that is not already an integer to integers.
		
		For example, in the 'features' map, we see that 'duration' is mapped to 0 while protocol_type
		is mapped to 1.

		It is important that these mappings are unique (no two labels map to the same integer).

		NOTE: This class and the data contained was manually created. This is important because it is 
		describing the data we care about. It is meta data.
		"""

		# A map of all features in the given data (What identifies a single row in the data 'table' -- think columns)
		features = {
			'duration': 0,
			'protocol_type': 1, 
			'service': 2, 
			'flag': 3, 
			'src_bytes': 4, 
			'dst_bytes': 5, 
			'land': 6, 
			'wrong_fragment': 7, 
			'urgent': 8, 
			'hot': 9, 
			'num_failed_logins': 10,
			'logged_in': 11,
			'num_compromised': 12, 
			'root_shell': 13, 
			'su_attempted': 14, 
			'num_root': 15, 
			'num_file_creations': 16, 
			'num_shells': 17, 
			'num_access_files': 18, 
			'num_outbound_cmds': 19, 
			'is_host_login': 20, 
			'is_guest_login': 21, 
			'count': 22, 
			'srv_count': 23, 
			'serror_rate': 24, 
			'srv_serror_rate': 25, 
			'rerror_rate': 26, 
			'srv_rerror_rate': 27, 
			'same_srv_rate': 28, 
			'diff_srv_rate': 29, 
			'srv_diff_host_rate': 30, 
			'dst_host_count': 31, 
			'dst_host_srv_count': 32, 
			'dst_host_same_srv_rate': 33, 
			'dst_host_diff_srv_rate': 34, 
			'dst_host_same_src_port_rate': 35, 
			'dst_host_srv_diff_host_rate': 36, 
			'dst_host_serror_rate': 37, 
			'dst_host_srv_serror_rate': 38, 
			'dst_host_rerror_rate': 39, 
			'dst_host_srv_rerror_rate': 40, 
			'training_label': 41, 						# This one was added to train our tree!
		}

		# A Map of all of the categories that the data might be classified as (remember that training label?)
		categories = {
			'normal.': 0, 
			'back.': 1, 
			'buffer_overflow.': 2, 
			'ftp_write.': 3, 
			'guess_passwd.': 4, 
			'imap.': 5, 
			'ipsweep.': 6, 
			'land.': 7, 
			'loadmodule.': 8, 
			'multihop.': 9, 
			'neptune.': 10, 
			'nmap.': 11,
			'perl.': 13, 
			'phf.': 14, 
			'pod.': 15, 
			'portsweep.': 16, 
			'rootkit.': 17, 
			'satan.': 18, 
			'smurf.': 19, 
			'spy.': 20, 
			'teardrop.': 21, 
			'warezclient.': 22, 
			'warezmaster.': 23, 
		}

		# Mapping of all protocol types to integers (protocols found by inspecting the data)
		protocol_types = {
			'tcp': 0, 
			'udp': 1, 
			'icmp': 2, 
		}

		# Mapping of all services to integers (services found by inspecting the data)
		services = {
			'http': 0, 
			'domain': 1, 
			'netbios_ssn': 2, 
			'urp_i': 3, 
			'Z39_50': 4, 
			'smtp': 5, 
			'gopher': 6, 
			'private': 7, 
			'echo': 8, 
			'printer': 9, 
			'red_i': 10, 
			'eco_i': 11, 
			'sunrpc': 12, 
			'ftp_data': 13, 
			'urh_i': 14, 
			'pm_dump': 15, 
			'pop_3': 16, 
			'pop_2': 17, 
			'systat': 18, 
			'ftp': 19, 
			'uucp': 20, 
			'whois': 21, 
			'netbios_dgm': 22, 
			'efs': 23, 
			'remote_job': 24, 
			'sql_net': 25, 
			'daytime': 26, 
			'ntp_u': 27, 
			'finger': 28, 
			'ldap': 29, 
			'netbios_ns': 30, 
			'kshell': 31, 
			'iso_tsap': 32, 
			'ecr_i': 33, 
			'nntp': 34, 
			'shell': 35, 
			'domain_u': 36, 
			'uucp_path': 37, 
			'courier': 38, 
			'exec': 39, 
			'tim_i': 40, 
			'netstat': 41, 
			'telnet': 42, 
			'rje': 43, 
			'hostnames': 44, 
			'link': 45, 
			'auth': 46, 
			'http_443': 47, 
			'csnet_ns': 48, 
			'X11': 49, 
			'IRC': 50, 
			'tftp_u': 51, 
			'imap4': 52, 
			'supdup': 53, 
			'name': 54, 
			'nnsp': 55, 
			'mtp': 56, 
			'bgp': 57, 
			'ctf': 58, 
			'klogin': 59, 
			'vmnet': 60, 
			'time': 61, 
			'discard': 62, 
			'login': 63, 
			'other': 64, 
			'ssh': 65, 

		}

		# Mapping of all flags to integers (Flags found by inspecting the data)
		flags = {
			'SF': 0, 
			'OTH': 1, 
			'RSTR': 2, 
			'S3': 3, 
			'S2': 4, 
			'S1': 5, 
			'S0': 6, 
			'RSTOS0': 7, 
			'REJ': 8, 
			'SH': 9, 
			'RSTO': 10, 

		}



	class Indices:
		duration = 0
		protocol_type = 1
		service = 2
		flag = 3
		src_bytes = 4
		dst_bytes = 5
		land = 6
		wrong_fragment = 7
		urgent = 8
		hot = 9
		num_failed_logins = 10
		logged_in = 11
		num_compromised = 12
		root_shell = 13
		su_attempted = 14
		num_root = 15
		num_file_creations = 16
		num_shells = 17
		num_access_files = 18
		num_outbound_cmds = 19
		is_host_login = 20
		is_guest_login = 21
		count = 22
		srv_count = 23
		serror_rate = 24
		srv_serror_rate = 25
		rerror_rate = 26
		srv_rerror_rate = 27
		same_srv_rate = 28
		diff_srv_rate = 29
		srv_diff_host_rate = 30
		dst_host_count = 31
		dst_host_srv_count = 32
		dst_host_same_srv_rate = 33
		dst_host_diff_srv_rate = 34
		dst_host_same_src_port_rate = 35
		dst_host_srv_diff_host_rate= 36
		dst_host_serror_rate = 37
		dst_host_srv_serror_rate = 38
		dst_host_rerror_rate = 39
		dst_host_srv_rerror_rate = 40
		training_label = 41

		all_index_labels = [
			'duration', 
			'protocol_type', 
			'service', 
			'flag', 
			'src_bytes', 
			'dst_bytes', 
			'land', 
			'wrong_fragment', 
			'urgent', 
			'hot', 
			'num_failed_logins', 
			'logged_in', 
			'num_compromised', 
			'root_shell', 
			'su_attempted', 
			'num_root', 
			'num_file_creations', 
			'num_shells', 
			'num_access_files', 
			'num_outbound_cmds', 
			'is_host_login', 
			'is_guest_login', 
			'count', 
			'srv_count', 
			'serror_rate', 
			'srv_serror_rate', 
			'rerror_rate', 
			'srv_rerror_rate', 
			'same_srv_rate', 
			'diff_srv_rate', 
			'srv_diff_host_rate', 
			'dst_host_count', 
			'dst_host_srv_count', 
			'dst_host_same_srv_rate', 
			'dst_host_diff_srv_rate', 
			'dst_host_same_src_port_rate', 
			'dst_host_srv_diff_host_rate', 
			'dst_host_serror_rate', 
			'dst_host_srv_serror_rate', 
			'dst_host_rerror_rate,' 
			'dst_host_srv_rerror_rate', 
		]

	class Labels:
		normal = 'normal.'
		back = 'back.'
		buffer_overflow = 'buffer_overflow.'
		ftp_write = 'ftp_write.'
		guess_passwd = 'guess_passwd.'
		imap = 'imap.'
		ipsweep = 'ipsweep.'
		land = 'land.'
		loadmodule = 'loadmodule.'
		multihop = 'multihop.'
		neptune = 'neptune.'
		nmap = 'nmap.'
		normal = 'normal.'
		perl = 'perl.'
		phf = 'phf.'
		pod = 'pod.'
		portsweep = 'portsweep.'
		rootkit = 'rootkit.'
		satan = 'satan.'
		smurf = 'smurf.'
		spy = 'spy.'
		teardrop = 'teardrop.'
		warezclient = 'warezclient.'
		warezmaster ='warezmaster.'

		all_labels = [
			'normal.', 
			'back.', 
			'buffer_overflow.', 
			'ftp_write.', 
			'guess_passwd.', 
			'imap.', 
			'ipsweep.', 
			'land.', 
			'loadmodule.', 
			'multihop.', 
			'neptune.', 
			'nmap.', 
			'normal.', 
			'perl.', 
			'phf.', 
			'pod.', 
			'portsweep.', 
			'rootkit.', 
			'satan.', 
			'smurf.', 
			'spy.', 
			'teardrop.', 
			'warezclient.', 
			'warezmaster.', 
		]


class DataParser(object):

	def __init__(self):


		# self.protocol_types = {}
		# self.services = {}
		# self.flags = {}

		self.data = []
		self.formatted_data = []
		self.training_data = []
		self.formatted_training_data = []
		self.file_path = 'data/kddcup.data_10_percent'
		self.load_file(self.file_path)
		self.format_data_for_tree()


	def load_file(self, file_path):
		"""
		Loads the training dataset and stores the category and the data separately.
		Assumes that the category for training is in the last index of the list.
		"""

		print('Loading data...')

		with open(file_path, 'r') as csv_file:
			reader = csv.reader(csv_file)

			count = 0

			for row in reader:
				# Split up the data (first 41 entries are data, the last one is a training label)
				self.data.append(row[:-1])
				self.training_data.append(row[-1])
				count = count + 1

				# test code (below) (To help identify features of the data, etc)

				# # Find all of the possible protocol types, services, and flags in the data given
				# DataFormatting.Mappings.protocol_types[row[DataFormatting.Mappings.features['protocol_type']]] = 1
				# DataFormatting.Mappings.protocol_types[row[DataFormatting.Mappings.features['service']]] = 1
				# DataFormatting.Mappings.protocol_types[row[DataFormatting.Mappings.features['flag']]] = 1
                #
				# # Show all of the data but only for the first 10 entries
				# print (row)
				# if count > 10:
				# 	break

				# end of test code

			print('Loaded ' + str(count) + ' rows.')

			# Prints out the found protocols, servces, and flags in the data
			print ('Protocols: ' + str(DataFormatting.Mappings.protocol_types.keys()))
			print ('Services: ' + str(DataFormatting.Mappings.services.keys()))
			print ('Flags: ' + str(DataFormatting.Mappings.flags.keys()))


	def format_data_for_tree(self):
		"""
		This function will format the data that we read in and prepare it for use in the 
		decision tree.

		Remember, the decision tree will ONLY use integers (no strings) so we have to swap 
		out the strings for their integer equivalents.

		This is where that big class 'DataFormatting.Mappings' comes in handy.
		To make this work, the mapped dictionary key is the string (data) and the value is the 
		integer that we want to swap it with.

		From inspection of the data, we know that only protocol_type, service, and flag are strings.
		'training_label' is a string too, but we are stripping that out when we load the file.

		From DataFormatting.Mappings.features:

		'protocol_type': 1, 
		'service': 2, 
		'flag': 3, 
		'training_label': 41, 
		"""

		print('Formatting data for decision tree...')

		prot_type_index = DataFormatting.Mappings.features['protocol_type']
		service_index = DataFormatting.Mappings.features['service']
		flag_index = DataFormatting.Mappings.features['flag']

		count = 0

		# Format the data we have been given
		for row in self.data:

			# Get the type first, then replace the data at that index with the mapped integer for that type
			protocol_type = row[prot_type_index]
			row[prot_type_index] = DataFormatting.Mappings.protocol_types[protocol_type]

			service = row[service_index]
			row[service_index] = DataFormatting.Mappings.services[service]

			flag = row[flag_index]
			row[flag_index] = DataFormatting.Mappings.flags[flag]

			self.formatted_data.append(row)
			count = count + 1

			# Debug: shows what is in the row as we go though it
			# print (row)
            #
			# if DataFormatting.Mappings.protocol_types[protocol_type] != 0:
			# 	print ('type = ' + str(protocol_type) + ', value = ' + str(DataFormatting.Mappings.protocol_types[protocol_type]))
			# 	print ('service = ' + str(service) + ', value = ' + str(DataFormatting.Mappings.services[service]))
			# 	print ('flag = ' + str(flag) + ', value = ' + str(DataFormatting.Mappings.flags[flag]))
			# 	print ('*'*40)
            #
			# if count > 10:
			# 	break

			# End DEBUG

		print ('Formatted ' + str(count) + ' rows.')

		print ('Formatting training data for decision tree...')

		count = 0
		for category in self.training_data:
			self.formatted_training_data.append(DataFormatting.Mappings.categories[category])
			count = count + 1

		print ('Formatted ' + str(count) + ' rows.')


if __name__ == "__main__":
	parser = DataParser()

	decision_tree = tree.DecisionTreeClassifier()

	print ('Training Decision Tree...')

	# Debug information showing what is in the fomatted lists
	# print '[DEBUG]\t#rows in formatted data: ' + str(len(parser.formatted_data))
	# print '[DEBUG]\t#rows in training data: ' + str(len(parser.formatted_training_data))

	decision_tree.fit(parser.formatted_data, parser.formatted_training_data)
	print ('Trained.')

	print ('Building graphical decision tree...')

	dot_data = StringIO()
	tree.export_graphviz(decision_tree, 
		out_file=dot_data, 
		feature_names=list(DataFormatting.Mappings.features.keys())[:-1],
		class_names=list(DataFormatting.Mappings.categories.keys()),
		filled=True,
		rounded=True, 
		special_characters=True
		)

	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf('IDS_Tree_Graph.pdf')

	print ('Done. Saved as IDS_Tree_Graph.pdf')
	print ('NOTE: Remember that we had to substitute integers for labels, so this graph may be hard to read.')


