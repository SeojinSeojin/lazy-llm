from src.utils.ezr import *
from src.language import load_model
from dotenv import load_dotenv
from src.prompts import load_prompt
from src.utils.results import save_results_txt
from graph import visualize 
import warnings
import time

def FEW(args, save_results = False, k = 1, model = None):
    warnings.filterwarnings("ignore")
    loaded_here = False
    if model == None:
        loaded_here = True
        model =  load_model(args)
    #random.seed(args.seed)
    records = []

    def _tile(lst, curd2h, budget):
        num = adds(NUM(),lst)
        n=100
        print(f"{len(lst):5} : {num.mu:5.3} ({num.sd:5.3})",end="")
        sd=int(num.sd*n/2)
        mu=int(num.mu*n)
        print(" "*(mu-sd), "-"*sd,"+"*sd,sep="")
        record = o(the = "result", N = len(lst), Mu = format(num.mu,".3f"), Sd = format(num.sd, ".3f"), Var = " "*(mu-sd) + "-"*sd + "+"*sd, Curd2h = format(curd2h, ".3f"), Budget = budget)
        records.append(record)

    def learner(i:data, callBack=lambda x,y,z:x):
        """
        
        """
        def _ranked(lst:rows, cur:row = None, budget:int = 0) -> rows:
            "Sort `lst` by distance to heaven. Called by `_smo1()`."
            lst = sorted(lst, key = lambda r:d2h(i,r))
            callBack([d2h(i,r) for r in lst], 0 if cur == None else d2h(i,cur), budget)
            # print(d2h of the best row)
            return lst

        def llm_guesser(current: row, done: rows) -> row:
            cut = int(.5 + len(done) ** 0.5)
            best = clone(i, done[:cut]).rows
            rest = clone(i, done[cut:]).rows
            best = [b[:len(i.cols.x)] for b in best]
            rest = [r[:len(i.cols.x)] for r in rest]
            base_prompt = load_prompt(args.dataset).getFewShot(best, rest, current[:len(i.cols.x)], cols=i.cols.x)
            prompt_variants = []

            # --- (1) No Ensemble: baseline ---
            if args.ensemble is None:
                text = model.generate_text(base_prompt)
                if "best" in text.lower():
                    return current
                return None

            # --- (2) Self-Consistency Ensemble: same prompt + minor variants ---
            elif args.ensemble == "self-consistency":
                votes = []
                num_reflections = 3

                # Step 1
                text1 = model.generate_text(base_prompt)
                votes.append(text1)

                # Step 2 - reflect on step 1
                reflection_prompt = base_prompt + [
                    {"role": "system", "content": f"Your previous reasoning: {text1}\nNow think step by step and re-evaluate your answer."}
                ]
                text2 = model.generate_text(reflection_prompt)
                votes.append(text2)

                # Step 3 - reflect on both previous steps
                reflection_prompt_2 = base_prompt + [
                    {"role": "system", "content": f"Previous reasoning attempts:\n1. {text1}\n2. {text2}\nNow make your final, consistent judgment."}
                ]
                text3 = model.generate_text(reflection_prompt_2)
                votes.append(text3)

                # --- Majority voting ---
                best_votes = sum(1 for t in votes if t and "best" in t.lower())
                if best_votes >= 2:
                    return current
                return None

            # --- (3) Heterogeneous Ensemble: different prompt templates ---
            elif args.ensemble == "heterogeneous":
                prompt_variants = [
                    base_prompt,
                    # Focus on quantitative reasoning
                    base_prompt + [{"role": "system", "content": "Focus on numeric differences and thresholds between examples."}],
                    # Focus on qualitative reasoning
                    base_prompt + [{"role": "system", "content": "Focus on qualitative patterns and logical rules rather than numeric values."}],
                    # Emphasize contrastive reasoning
                    base_prompt + [{"role": "system", "content": "Compare best and rest examples explicitly before making a decision."}]
                ]
                votes = []

                for p in prompt_variants:
                    text = model.generate_text(p)
                    if "best" in text.lower():
                        votes.append(1)
                    else:
                        votes.append(0)

                # --- Voting decision (same output type as before) ---
                if sum(votes) >= (len(votes) / 2):
                    return current
                return None
            
        
        def _smo1(todo:rows, done:rows) -> rows:
            "Guess the `top`  unlabeled row, add that to `done`, resort `done`, and repeat"
            count = 0
            for idx, k in enumerate(todo, start=1):
                start_time = time.time()
                if len(done) >= args.last:
                    break
                top = llm_guesser(k, done)
                if top is None:
                    continue
                btw(d2h(i, top))
                done += [top]
                done = _ranked(done, top, count)
                elapsed_min = (time.time() - start_time) / 60
                print(f"[TIMER] step {idx}/{args.last} took {elapsed_min:.2f} min")
                count += 1
            return done


        # todo: sampler, sample ratio to args
        i_sampled = random.choices(i.rows, k=k)
        todo_full = i_sampled[args.label:]
        sample_ratio = 0.2
        sample_size = min(max(1, int(len(todo_full) * sample_ratio)), 100)
        todo_subset = random.sample(todo_full, sample_size)

        print(f"[INFO] Sampling {sample_size}/{len(todo_full)} rows for this run.")
        return _smo1(todo_subset, _ranked(i_sampled[:args.label]))

    
    if(save_results):
        results = learner(DATA(csv(args.dataset)), _tile)
        save_results_txt(model = args.model + "_" + args.llm, dataset = args.dataset, records =  records)
        time.sleep(5)
        visualize(dataset = args.dataset[args.dataset.rfind('/')+1:-4], show = 'All', save_fig= True, display = False)
    else: results = learner(DATA(csv(args.dataset)))
    if loaded_here:
        model.unload_model()
    return results