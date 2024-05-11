# Latent Drift Correction #
### extension for Forge webui for Stable Diffusion ###

---
## Install ##
Go to the **Extensions** tab, then **Install from URL**, use the URL for this repository.

--- 
![](screenshot.png "image of extension UI")

---
## Basic usage ##
Pick methods (seems not much difference, might remove some after more testing).

---
## Advanced / Details ##
Delaying the start can be beneficial, as can early ending.
This sort of correction has a tendency to prevent extremes of lighting.
custom functions:
* M: mean
* m: median
* q(n): quantile. 0.5 is same as median. High values for low-light scenes.
* rM(n, m): mean of range, rM(0, 0.5) gives mean of lowest 50%


---
## To do? ##
25/04/2024: added saving/loading of custom functions

05/11/2024: bug with centered mean when topK = 0.5: (0.5-0.5 = 0, who knew?)

---
## License ##
Public domain. Unlicense. Free to a good home.
All terrible code is my own. Use at your own risk, read the code.

---
## Credits ##
General idea from (Birch Labs)[https://birchlabs.co.uk/machine-learning#combating-mean-drift-in-cfg] but this is after CFG

SoftClamp method by (Timothy Alexis Vass)[https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space]


---
