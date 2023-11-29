import json
import pandas as pd

def postproc(args, section, state_epoch, correct_flag):
    if args.cfg in ['Salesforce/_T5_large_finetune_squall.cfg', '_Omnitab_large_finetune_squall_tableqa.cfg']:
        filter = ["query_tokens", 
                  "json_path", 
                  "db_path", 
                  "db_table_names", 
                  "db_column_names", 
                  "header",
                  "db_primary_keys", 
                  "db_foreign_keys", 
                  "db_column_types"]
        with open(f"./{args.output_dir}/predictions_{section}_{state_epoch}.json", "r") as f:
            data = json.load(f)
        to_save = []
        for i, ex in enumerate(data):
            sample={}
            sample['correct_flag']=correct_flag[i]
            for key in ex:
                if key not in filter:
                    sample[key]=ex[key]
            to_save.append(sample)
        with open(f"./{args.output_dir}/predictions_{section}_{state_epoch}.json", "w") as json_file:
            json.dump(to_save, json_file, indent=4)
        
        df = pd.DataFrame(to_save)
        df.to_csv(f"./{args.output_dir}/predictions_{section}.csv")