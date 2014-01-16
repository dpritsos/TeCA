
import tables as tb

res_h5file = tb.openFile('/home/dimitrios/Synergy-Crawler/SANTINIS/SANTINIS_Words_RFSE_Part-1.h5', 'r')

for nds in res_h5file.walkGroups('/'):

	try:
		ar = res_h5file.getNode(nds, name='docs_term_counts')
		print ar
	except:
		pass
		

res_h5file.close()

print 'Done'