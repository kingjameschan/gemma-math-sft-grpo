from trl import GRPOConfig
import inspect
sig = inspect.signature(GRPOConfig.__init__)
gen_params = [p for p in sig.parameters if any(k in p.lower() for k in ['token', 'generat', 'max_', 'temper', 'beta', 'group'])]
print("Gen-related params:", gen_params)
