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

## References

- [Model Context Protocol (MCP) Documentation](https://github.com/ModelContextProtocol/spec)
- [UV - Python Package and Project Manager](https://github.com/astral-sh/uv)
- [MCP Server Implementation Guide](https://github.com/ModelContextProtocol/mcp-server)
- [MCP Client Libraries](https://github.com/ModelContextProtocol/mcp-clients)
- [Python HTTPX - HTTP client for Python](https://www.python-httpx.org/)

For more information about MCP and its ecosystem, please visit the [official MCP GitHub organization](https://github.com/ModelContextProtocol).