# Configuration Information of JABAS

## User-defined parameters (optional)
### Adaptive Batching
```
batch_size_adjust_rate
```
The adaptive rate of current global batch size.

The range of value is between 0 and 1 where 1 means 100%. Default value is 0.1 (10%).

```
batch_size_adjust_interval
```
The interval (step) of computing gradient similarity and adjust global batch size.
Default value is 100.

### Automatic Scaling
```
resource_alloc_unit
```
The unit of number of resource allocation. Default value is '4gpu'. Minimum value is '2gpu'.
```
homo_servers
```
Ths list of homogeneous GPU servers.
If multiple nodes of homogeneous GPUs exist, this configuration helps reduce the number of candidate resource allocations for auto-scaling. Default value is [].

Example is ```[["node0", "node1"], ["node2", "node3"]]```.

## Utility
### Memory profile data
Visualize the local batch size and maximum number of VSWs on each type of GPU.
```
python -m jabas.profiler.api.memory_profile_data_summary -d [memory profile dir]
```
