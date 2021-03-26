# GPU powered Blockchain node
This project aims to create a software which acts as a Blockchain Full Node and also as a Blockchain Miner. Leveraging GPU's SIMD architecture, we aim to create an efficient way to verify transactions, verify candiate blocks and also search the blockchain data.

## Background
Blockchain has been used in many applications, most notably are cryptocurrency and supply chain management. A blockchain consists of data sets which are composed from a chain of blocks, where one block contains multiple transaction<sup>[1]</sup>. When a block is added, it contains a timestamp of when the block is constructed, hash to the previous block, and a nonce that will help in making a valid block. The hash to the previous block guarantees the contiguous chain of blocks in the blockchain. The hash itself depends on the transactions inside the block, so if one transaction changes, the whole hash changes, meaning the blockchain must be chaged.<br>
Maintaining a blockchain however, is an expensive and time-consuming task. Looking at the cryptocurrency example, we know the news about people buying tens of GPU in order to mine and maintain the main cryptocurrency blockchain. For example, in creating a new block to be added into the main blockchain, a hash must be within the difficulty threshold. In order to do so, a nonce must be used in order to alter the hash output. Hashing a block is an expensive and time-consuming algorithm, especially when the difficulty bar is set to very low. The second example is verifying a valid transaction. Verifying a transaction is a much cheaper algorithm compared to hashing, however doing a serialized transaction verifying algorithm when there are millions of transaction is not a feasible idea.<br>
In blockchain, there are what's called nodes. Nodes are the backbone of a blockchain, they store, spread and preserve the blockchain data. There are full nodes and partial nodes. Full nodes are the real backbone of the blockchain, they do all the process of verifying blocks which will be adde and preserving the blockchain data and serving the data when requested (searching the blockchain). While Partial nodes are nodes which trust one of the few Full node and do a simplified version of the verification. While Mining Process is necessary for a blockchain to work, Full nodes are also important. But in reality, An estimated 1 million bitcoin (one of the largest blockchain) miners are in operation, while only around 11 thousand full nodes are operational in the bitcoin blockchain. This is where GPU comes in. GPU is the backbone of blockchain. The SIMD architecture of GPU is perfect from verifying transactions, to hashing a block, since there are only Single Instruction (SI) in hashing and verifying transactions. With this, our group is interested in creating a parallelized algorithm in blockchain management.<br>

## Challenges
There are many mining software which leverage GPU, but for a full node, there is only one work which leverages GPU to for the node<sup>[2]</sup>. That previous work only focuses on parallelizing the search queries in the full nodes. This project aims to parallelize the mining algorithm, the candidate block verification, and also search queries. While the mining algorithm would be embarassingly parallelizable, the candidate block verification and search queries are not. The challenges of this project lies in the creation of those two algorithm, The CPU version of the algorithm isn't that simple, and there aren't any previous work which we can look at (for the candidate block verification), and the existing work on the search queries uses quite a complex method. And tying it all together into one cohesive software would also be a challenge.


## Deliverables
The end product would be a program which will run as a miner if idle, verify candidate blocks when a new block is created, and also handle search queries. We would create 3 GPU modules:
1. Mining Algorithm
2. Blockchain Search Queries
3. Candidate Block Verification

Garry will create module 1 and 2, while Kevin Fernaldy will create module 3.
## Timeline
| Week No | Deadline Date | Work                                                                                    |
|---------|---------------|-----------------------------------------------------------------------------------------|
| 1       | 6 April       | CPU Proof of Concept, GPU version of the Mining algorithm                               |
| 2       | 13 April      | Minimum Working Solution of the Module 2 and Module 3, An optimal solution for Module 1 |
| 3       | 20 April      | Optimizing Module 2 and Module 3                                                        |

## Source
[1] Nofer, M., Gomber, P., Hinz, O. et al. Blockchain. Bus Inf Syst Eng 59, 183–187 (2017)

[2] Morshima, S., Matsutani, H. Accelerating Blockchain Search of Full Nodes Using GPUs (2018)

