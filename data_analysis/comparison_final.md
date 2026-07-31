# Final 2x2 comparison: {GAN, Diffusion} x {free, constructed}
mean +/- std over seeds x held-out days 8/9/10

| paradigm | mode | seeds | valid_all | spread_err | depth_err | nn_dist | diversity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GAN | free | 3 | 0.0001 +/- 0.0001 | 0.0003 +/- 0.0001 | 0.0175 +/- 0.0095 | 0.0789 +/- 0.0105 | 0.5392 +/- 0.0035 |
| GAN | constructed | 3 | 1.0000 +/- 0.0000 | 0.0001 +/- 0.0000 | 0.0170 +/- 0.0077 | 0.0783 +/- 0.0116 | 0.5392 +/- 0.0045 |
| Diffusion | free | 3 | 0.0000 +/- 0.0000 | 0.0006 +/- 0.0003 | 0.0252 +/- 0.0209 | 0.0779 +/- 0.0107 | 0.5304 +/- 0.0070 |
| Diffusion | constructed | 3 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0254 +/- 0.0191 | 0.0923 +/- 0.0094 | 0.5371 +/- 0.0054 |