

##  MCP init

```bash

uv init project-name

uv add mcp[cli] httpx

```

### RUN mcp server
```bash
mcp dev main.py


### install to  Claude Config
mcp install main.py

### Test in MCP Inspector
mcp dev main.py
```






## Weather Server MCP

### Run MCP
```bash

uv --directory /Users/tharhtet/Documents/github/ML-in-Prod-batch-2/13_Adv_AI_methods/MCP/weather-server-python run main.py

```


### Weather Server MCP Config for Claude
```json
 "weather-server-python": {
            "command": "/opt/anaconda3/envs/ths_dev/bin/uv",
            "args": [
                "--directory",
                "/Users/tharhtet/Documents/github/ML-in-Prod-batch-2/13_Adv_AI_methods/MCP/weather-server-python",
                "run",
                "main.py"
            ]
        }
  
  }
```