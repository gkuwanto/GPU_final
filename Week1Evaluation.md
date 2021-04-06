# Week 1 Progress
- Implemented CPU version of transaction verfication
- Implemented CPU version of mininig algorithm
- Created simple blockchain.
## Analysis
- After trying to implement the search algorithm, we concluded that we needed to create a key value store database which would be to complex for the project scope.
- And after implementing the CPU version, while the mining is embarassingly parallelizable, we found that the implementation isn't as easy as it sounds and has a few ways to be optimized
- For the CPU version we used several libraries which need to be re-implemented into a GPU friendly version. which would make 3 modules to big of a scope for the project.

## Conclusion
- We decided to reduce the project scope to 2 GPU modules, the verfication module and the mining module. Kevin will work on the verification module while Garry will work on the mining module.
- We concluded that we need to impelement a new database to create the search module, would be to big of a scope for the project.
- After a weeks work, we updated our timeline to:
### Updated Timeline
| Week No | Deadline Date | Work                                                                                    |
|---------|---------------|-----------------------------------------------------------------------------------------|
| 1       | 6 April       | CPU Demo                      |
| 2       | 13 April      | Minimum working GPU Version of the Mining Module and Transaction / Candidate Block verification |
| 3       | 20 April      | Optimizing the created kernels              |