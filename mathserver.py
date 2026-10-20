from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")  #"Math" → name you give your server

@mcp.tool()

def add(a:int , b:int)-> int:
    """ _summary_ add to numbers """
    return a + b

@mcp.tool()

def multiply(a:int , b:int)-> int:
    """ multiply a and b """
    return a*b

# the transport="stdio" argument tells the server to:
# use the standarad input/output(stdin /stdout) tp receive and respond to tool function calls

if __name__== "__main__":
    mcp.run(transport="stdio")