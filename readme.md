# Inspiration


[Executorch extended with modules](https://pytorch.org/executorch/stable/extension-module.html) - more doc on building information in [Intro to LLMs in Executorch](https://github.com/pytorch/executorch/blob/main/docs/source/llm/getting-started.md)

### Step 1

Generate the .pte model (`python simple_conv.py`)


### Step 2
Build and run the C++ program
```
(rm -rf cmake-out && mkdir cmake-out && cd cmake-out && cmake ..)
```

```
cmake --build cmake-out -j10;cmake-out/webcam_executorch
```

