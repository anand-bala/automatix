# Monitoring with STREL Automata

To run an example trace, run:

```bash
python ./monitoring_example.py \
    --spec "<spec_file>.py" \
    --map "<map_file>.json" \
    --trace "<trace_file>.csv" \
    # --ego "drone_03" \ # Skip to monitor from all locations, or specify multiple times. \
    # --timeit # Do rigorous profiling \
    # --online # Do online monitoring \
```

- The files, `./establish_comms_spec.py` and `./reach_avoid_spec.py` , are specification
  files (can be used to substitute the `<spec_file>.py` field above.

- Directories `./data/` contain the map files (named `./data/Map_<n>/map_params.json`)
  and trace files (CSV files under each directory).

## Generating more traces

The code in `./swarmlab/` is used to generate trajectories. The main file is
`./swarmlab/example_simulation.m`, which can be edited to point to various maps and
drone configurations (see the bottom of the file). Each map file is one of the
`param_map_<num>.m`, which can be edited to make new maps.

Before running the simulations, ensure that you have MATLAB installed and the
[Swarmlab](https://github.com/lis-epfl/swarmlab) package added in the [MATLAB search
path](https://www.mathworks.com/help/matlab/search-path.html).
