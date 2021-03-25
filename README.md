# GPU powered Blockchain node
1-3 sentence summary

## Background
Blockchain has been used in many applications, most notably are cryptocurrency and supply chain management. A blockchain consists of data sets which are composed from a chain of blocks, where one block contains multiple transaction[1]. When a block is added, it contains a timestamp of when the block is constructed, hash to the previous block, and a nonce that will help in making a valid block. The hash to the previous block guarantees the contiguous chain of blocks in the blockchain. The hash itself depends on the transactions inside the block, so if one transaction changes, the whole hash changes, meaning the blockchain must be chaged.<br>
Maintaining a blockchain however, is an expensive and time-consuming task. Looking at the cryptocurrency example, we know the news about people buying tens of GPU in order to mine and maintain the main cryptocurrency blockchain. For example, in creating a new block to be added into the main blockchain, a hash must be within the difficulty threshold. In order to do so, a nonce must be used in order to alter the hash output. Hashing a block is an expensive and time-consuming algorithm, especially when the difficulty bar is set to very low. The second example is verifying a valid transaction. Verifying a transaction is a much cheaper algorithm compared to hashing, however doing a serialized transaction verifying algorithm when there are millions of transaction is not a feasible idea.<br>
This is where GPU comes in. GPU is the backbone of blockchain. The SIMD architecture of GPU is perfect from verifying transactions, to hashing a block, since there are only Single Instruction (SI) in hashing and verifying transactions. With this, our group is interested in creating a parallelized algorithm in blockchain management.

## Challenges
1-2 paragraph explaining what will be tricky to implement

## Deliverables
1 paragraph explaining endproduct, what GPU Modules to be made and also indicating who will make which GPU modules
## Timeline
Weekly breakdown of what each person will do and what is to be expected

## Source
[1] Nofer, M., Gomber, P., Hinz, O. et al. Blockchain. Bus Inf Syst Eng 59, 183–187 (2017)
