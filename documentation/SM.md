# Single Model

1. Test 1
- Environment: Mitarbeiter cluster, driver memory 5g, executor memory 21 g
- On fold 1, LDA topics:150, Peer papers:1, Pairs generation: edp (equally distributed pairs)
- Steps
-- fit LDA (44s)
-- LDA training (15 mins)
-- fit LTR (44s)

- Fold 1 information

-- paper corpus 
Data Distribution on 9 Executors:
1) Node - 3 46.2 KB (On Heap Memory Usage)
2) Node 5 - 897.6 KB
2 partitions:
Node 5 - 897.6 KB (Size in memory)
Node 3 - 446.2 KB

Cached Partitions: 2
Total Partitions: 2
Memory Size: 943.8 KB
Disk Size: 0.0 B 

Because paper corpus is partioned only on 2 nodes. LDA training is done only by 2 executors/nodes.
-- training data
Data Distribution on 9 Executors:
dbisma06 - 122.9 KB (11.0 GB Remaining) 
dbisma04 - 38.9 KB (11.0 GB Remaining) 

4 partitions:
104.3 KB - dbisma06.informatik.privat:45070
38.9 KB - dbisma04.informatik.privat:34997
18.5 KB - dbisma06.informatik.privat:45070
5.7 KB - dbisma09.informatik.privat:35732
Cached Partitions: 4
Total Partitions: 4
Memory Size: 167.6 KB 

- In total time taken: 
- Evaluation received: 30 mins without saving overall results

- When adding LDA topics to training set, data becomes 4.1GB
