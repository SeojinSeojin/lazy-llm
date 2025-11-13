import os
from src.utils.config import parse_arguments
from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize,visualize2 
import warnings
import time
from src.models.few import FEW
from src.models.smo import SMO
#from src.models.zero import ZERO
from src.models.warm_start import warm_smo
from src.models.gpm import gpms
from src.models.tpe import TPE
from src.models.diffevol2 import diffevols


def explore(B, R):
    return (B + R) / (abs(B - R) + 1E-30)

def exploit(B, R):
    return B

def norm(x, shift):
    min_x = min(x)
    max_x = max(x)
    normalized = [(xi - min_x) / (max_x - min_x) + shift for xi in x]
    return normalized

def m(i, n, shift):
    exp_value = math.exp(0.25 * i)
    exp_values = [math.exp(0.25 * j) for j in range(n)]
    normalized_values = norm(exp_values, shift)
    return normalized_values[i]

def fews(args):
    print("=== FEWS STARTED ===")

    e = math.exp(1)
    policies = dict(
        exploit=lambda B, R: B - R,
        EXPLORE=lambda B, R: (e**B + e**R) / abs(e**B - e**R + 1E-30)
    )
    repeats = 20
    d = DATA(csv(args.dataset))
    rxs = {}

    rxs["baseline"] = SOME(
        txt=f"baseline,{len(d.rows)}",
        inits=[d2h(d, row) for row in d.rows]
    )

    rx = f"rrp,{int(0.5 + math.log(len(d.rows), 2) + 1)}"
    rxs[rx] = SOME(txt=rx)
    for _ in range(repeats):
        best, _, _ = branch(d, d.rows, 4)
        rxs[rx].add(d2h(d, best[0]))

    tests = [
        #{"name": "phi3-mini", "lasts" : [20, 30], "repeats" : 10}, 
        #{"name": "llama3-8b", "lasts" : [20, 30], "repeats" : 10}, 
        #{"name" : "phi3-medium", "lasts" : [15], "repeats" : 1},
        # {"name": "gemini", "lasts": [10], "repeats": 3, "ensemble": None},
        # {"name": "gemini", "lasts": [10], "repeats": 3, "ensemble": "self-consistency"},
        # {"name": "gemini", "lasts": [10], "repeats": 3, "ensemble": "heterogeneous"},
        {"name": "phi3-mini", "lasts" : [20], "repeats" : 10, "ensemble": None}, 
        {"name": "phi3-mini", "lasts" : [20], "repeats" : 10, "ensemble": "self-consistency"}, 
        {"name": "phi3-mini", "lasts" : [20], "repeats" : 10, "ensemble": "heterogeneous"}, 
    ]

    print("Configured tests:", tests)
    k = min(500, len(d.rows))

    for test in tests:
        args.llm = test["name"]
        args.ensemble = test["ensemble"]

        # Load model once per configuration
        model = load_model(args)
        print(f"[LOAD] Model loaded: {model.model_name}")

        # Model info (works for both Local_LLM and API_LLM)
        model_info = model.get_params() if hasattr(model, "get_params") else "N/A"
        print(f"[INFO] Model info: {model_info}")

        for last in test["lasts"]:
            args.last = last
            rx_name = f"llm_{args.llm}({model_info}),{last},ensemble-{args.ensemble}"
            rxs[rx_name] = SOME(txt=rx_name)

            for _ in range(test["repeats"]):
                print(f"[RUN] {rx_name}")
                try:
                    # FEW() internally uses model.generate_text()
                    result = FEW(args, save_results=False, k=k, model=model)[0]
                    rxs[rx_name].add(d2h(d, result))
                except Exception as e:
                    print(f"[ERROR] FEW() failed: {type(e).__name__}: {e}")
                    continue

        # Clean up (works for both API_LLM and Local_LLM)
        if hasattr(model, "unload_model"):
            model.unload_model()

    scoring_policies = [
        ("exploit", lambda B, R, I, N: exploit(B, R)),
        ("explore", lambda B, R, I, N: explore(B, R)),
        ("SimAnneal", lambda B, R, I, N: abs(((B + 1) ** m(I, N, 1) + (R + 1)) / (abs(B - R) + 1E-30))),
    ]

    for last in [20, 25, 30, 35, 40, 45, 50, 55, 60]:
        the.Last = last
        guess = lambda: clone(d, random.choices(d.rows, k=last), rank=True).rows[0]
        rx = f"random,{last}"
        rxs[rx] = SOME(txt=rx, inits=[d2h(d, guess()) for _ in range(repeats)])
        for guessFaster in [True]:
            for what, how in scoring_policies:
                the.GuessFaster = guessFaster
                rx = f"{what},{the.Last}"
                rxs[rx] = SOME(txt=rx)
                for _ in range(repeats):
                    btw(".")
                    rxs[rx].add(d2h(d, smo(d, how)[0]))
                btw("\n")

    # --------------------------------------------
    # 7️⃣ Generate report
    # --------------------------------------------
    report(rxs.values())
    print("=== FEWS FINISHED ===")


# def zeros(args):
#     "try different sample sizes"
#     policies = dict(exploit = lambda B,R: B-R,
#                     EXPLORE = lambda B,R: (e**B + e**R)/abs(e**B - e**R + 1E-30))
#     repeats=20
#     d = DATA(csv(args.dataset))
#     e = math.exp(1)
#     rxs={}
#     rxs["baseline"] = SOME(txt=f"baseline,{len(d.rows)}",inits=[d2h(d,row) for row in d.rows])
#     rx=f"rrp,{int(0.5+math.log(len(d.rows),2)+1)}"
#     rxs[rx] = SOME(txt=rx)
#     for _ in range(repeats):
#         best,_,_ = branch(d,d.rows,4); rxs[rx].add(d2h(d,best[0]))

#     tests = [
#         {"name": "phi3-mini", "lasts" : [20, 30, 40], "repeats" : 5},
#         {"name": "llama3-8b", "lasts" : [20, 30], "repeats" : 5},
#         #{"name" : "phi3-medium", "lasts" : [15], "repeats" : 1}
#     ]
    
#     k = min(500, len(d.rows))
#     for test in tests:
#         args.llm = test["name"]
#         (model, dir) =  load_model(args).get_pipeline()
#         for last in test["lasts"]:
#             rx =f"llm_{args.llm}({load_model(args).get_params(model)}),{last}"
#             rxs[rx] = SOME(txt= rx)
#             for _ in range(test["repeats"]):
#                 args.last = last
#                 rxs[rx].add(d2h(d, ZERO(args, False, k, model)[0]))
#         unload_model(model, dir)

    

    
#     scoring_policies = [
#         ('exploit', lambda B, R, I, N: exploit(B, R)),
#         ('explore', lambda B, R, I, N: explore(B, R)),
#         #('b2', lambda B, R, I, N: (B**2) / (R + 1E-30)),
#         ('SimAnneal', lambda B, R, I, N: abs(((B + 1) ** m(I, N, 1) + (R + 1)) / (abs(B - R) + 1E-30))),
#         #('ExpProgressive', lambda B, R, I, N: m(I, N, 0) * exploit(B,R) + (1 - m(I, N, 0)) * explore(B,R))
#     ]
            
    
#     # rx =f"llm,phi3-medium,15"
#     # args.llm = 'phi3-medium'
#     # rxs[rx] = SOME(txt= rx)
#     # args.last = 15
#     # rxs[rx].add(d2h(d, vanilla1(args, False, k)[0]))

    
#     for last in [20,25,30,35,40,45,50,55,60]:
#       the.Last= last
#       guess = lambda : clone(d,random.choices(d.rows, k=last),rank=True).rows[0]
#       rx=f"random,{last}"
#       rxs[rx] = SOME(txt=rx, inits=[d2h(d,guess()) for _ in range(repeats)])
#       for  guessFaster in [True]:
#         for what,how in  scoring_policies:
#           the.GuessFaster = guessFaster
#           rx=f"{what},{the.Last}"
#           rxs[rx] = SOME(txt=rx)
#           for _ in range(repeats):
#              btw(".")
#              rxs[rx].add(d2h(d,smo(d,how)[0]))
#           btw("\n")
      
#     report(rxs.values())

def warms(args):

    repeats=20
    d = DATA(csv(args.dataset))
    e = math.exp(1)
    rxs={}
    rxs["baseline"] = SOME(txt=f"baseline,{len(d.rows)}",inits=[chebyshev(d,row) for row in d.rows])
    rx=f"rrp,{int(0.5+math.log(len(d.rows),2)+1)}"
    rxs[rx] = SOME(txt=rx)
    for _ in range(repeats):
        best,_,_ = branch(d,d.rows,4); rxs[rx].add(chebyshev(d,best[0]))

    scoring_policies = [
        ('exploit', lambda B, R, I, N: exploit(B, R)),
        ('explore', lambda B, R, I, N: explore(B, R)),
        #('b2', lambda B, R, I, N: (B**2) / (R + 1E-30)),
        #('Focus', lambda B, R, I, N: abs(((B + 1) ** m(I, N, 1) + (R + 1)) / (abs(B - R) + 1E-30))),
        #('ExpProgressive', lambda B, R, I, N: m(I, N, 0) * exploit(B,R) + (1 - m(I, N, 0)) * explore(B,R))
    ]
    
    for last in [20,25,30]:
      args.last= last
      guess = lambda : clone(d,random.choices(d.rows, k=last),rank=True).rows[0]
      rx=f"random,{last}"
      rxs[rx] = SOME(txt=rx, inits=[chebyshev(d,guess()) for _ in range(repeats)])
      
      gps = ['UCB_GPM', 'PI_GPM', 'EI_GPM']
      samplers = ['random', 'distkpp', 'dpp', 'kcenter'] # better use ['random'] 
      for guesFaster in [True]:
        for what in gps:
            for sampler in samplers:
                rx = f"{what},{sampler},{last}"
                rxs[rx] = SOME(txt=rx)
                for _ in range(repeats):
                    btw(".")
                    rxs[rx].add(chebyshev(d, gpms(args, what, sampler)[0]))
                btw("\n")
      for guesFaster in [True]:
        for sampler in samplers:
            rx = f"TPE,{sampler},{last}"
            rxs[rx] = SOME(txt=rx)
            for _ in range(repeats):
                btw(".")
                rxs[rx].add(TPE(args, sampler))
            btw("\n")
      
      graphs = {'exploit' : [], 'LINEAR/exploit' : [], 'LLM/exploit' : []}

      for  guessFaster in [True]:
        for what,how in  scoring_policies:
          the.GuessFaster = guessFaster
          rx=f"{what},{args.last}"
          rxs[rx] = SOME(txt=rx)
          for _ in range(repeats):
            btw(".")
            res,data = smo(d,how)
            rxs[rx].add(chebyshev(d,res[0]))
            if last == 20 and what in graphs.keys() : graphs[what].append(data)
        btw("\n")

      for  guessFaster in [True]:
        for start in ['gemini']:#['gemini','gpt']:
            args.llm = start
            for what,how in  scoring_policies:
                the.GuessFaster = guessFaster
                rx=f"{start}/{what},{args.last}"
                rxs[rx] = SOME(txt=rx)
                for _ in range(repeats):
                    btw(".")
                    #time.sleep(10)
                    if start == 'LLM' and len(d.rows) < 50:
                        res,data = smo(d,how) # this heuristic works because LLM warm start performs poorly across all small datasets
                        rxs[rx].add(chebyshev(d,res[0]))
                    else :
                        if args.llm == 'gpt': time.sleep(10)
                        res, data = warm_smo(args,how,method = start)
                        rxs[rx].add(chebyshev(d,res[0]))
                    if last == 20 and f'{start}/{what}' in graphs.keys(): graphs[f'{start}/{what}'].append(data)
            btw("\n")
    
    for last in [20,25,30,100]:
        args.last= last
        for guesFaster in [True]:
            args.last = last
            rx = f"DEHB,{last}"
            rxs[rx] = SOME(txt=rx)
            for _ in range(repeats):
                btw(".")
                rxs[rx].add(diffevols(args)[0])
            btw("\n")
    



       


    report(rxs.values())

    if args.graph:
        "plot the graph depending on the config"
        for n, _ in graphs.items():
            print(n, len(_))
        visualize2(args.dataset, graphs)


    


        
if __name__ == "__main__":
    # print("main")
    load_dotenv()
    # print("dotenv loaded: ", os.environ)
    args = parse_arguments()
    # print("args: ",args)
    if(args.model == 'few'):
        fews(args)
    # if(args.model == 'smo'):
    #     SMO(args)
    # if(args.model == 'fews'):
    #     fews(args)
    # if(args.model == 'zero'):
    #     ZERO(args,True)
    # if(args.model == 'zeros'):
    #     zeros(args)

    if(args.model == 'warms'):
        warms(args)

    #print(save_results())



