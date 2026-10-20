from mcp.server.fastmcp import FastMCP
 
mcp=FastMCP("Weather")  #"Weather" → name you give your server

@mcp.tool()

async def get_weather(location:str)-> str:
    """get the the live weather info for the city """
    return f"it is always mist in {location}"

if __name__=="__main__":
    mcp.run(transport="streamable-http")