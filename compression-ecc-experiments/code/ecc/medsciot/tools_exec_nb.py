"""Execute a notebook in place with the 'proj' kernel and store outputs."""
import json, sys, os
from jupyter_client.manager import start_new_kernel
path = sys.argv[1]
nb = json.load(open(path))
km, kc = start_new_kernel(kernel_name=nb["metadata"]["kernelspec"]["name"], cwd=os.path.dirname(os.path.abspath(path)))
import os; os.chdir(os.path.dirname(os.path.abspath(path)))
kc.execute("%matplotlib inline"); kc.get_shell_msg(timeout=60)
n = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    n += 1
    msg_id = kc.execute(cell["source"])
    outs = []
    while True:
        m = kc.get_iopub_msg(timeout=600)
        if m["parent_header"].get("msg_id") != msg_id:
            continue
        t, c = m["msg_type"], m["content"]
        if t == "stream":
            outs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
        elif t in ("display_data", "execute_result"):
            o = {"output_type": t, "data": c["data"], "metadata": c.get("metadata", {})}
            if t == "execute_result":
                o["execution_count"] = n
            outs.append(o)
        elif t == "error":
            outs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"], "traceback": c["traceback"]})
            print("ERROR in cell", n, c["ename"], c["evalue"])
        elif t == "status" and c["execution_state"] == "idle":
            break
    cell["outputs"], cell["execution_count"] = outs, n
kc.stop_channels(); km.shutdown_kernel()
json.dump(nb, open(path, "w"), indent=1)
print("executed", n, "code cells")
