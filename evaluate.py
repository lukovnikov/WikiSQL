#!/usr/bin/env python
import json, os, re, glob
from argparse import ArgumentParser
from tqdm import tqdm
from lib.dbengine import DBEngine
from lib.query import Query
from lib.common import count_lines
import numpy as np

#if __name__ == '__main__':
#    parser = ArgumentParser()
#    parser.add_argument('source_file', help='source file for the prediction')
#    parser.add_argument('db_file', help='source database for the prediction')
#    parser.add_argument('pred_file', help='predictions by the model')
#    args = parser.parse_args()

def eval_one(db_file, pred_file, source_file):
    engine = DBEngine(db_file)
    exact_match = []
    with open(source_file) as fs, open(pred_file) as fp:
        grades = []
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(source_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg['sql'])
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            pred = ep['error']
            qp = None
            if not ep['error']:
                try:
                    qp = Query.from_dict(ep['query'])
                    pred = engine.execute_query(eg['table_id'], qp, lower=True)
                except Exception as e:
                    pred = repr(e)
            correct = pred == gold
            match = qp == qg
            grades.append(correct)
            exact_match.append(match)
        result = {
            'ex_accuracy': sum(grades) / len(grades),
            'lf_accuracy': sum(exact_match) / len(exact_match),
            }
        return result


def eval_one_qelos(db_file, pred_file, source_file):
    engine = DBEngine(db_file)
    exact_match = []
    with open(source_file) as fs, open(pred_file) as fp:
        grades = []
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(source_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg['sql'])
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            qp = None
            try:
                qp = Query.from_dict(ep)
                pred = engine.execute_query(eg['table_id'], qp, lower=True)
            except Exception as e:
                pred = repr(e)
            correct = pred == gold
            match = qp == qg
            grades.append(correct)
            exact_match.append(match)
        result = {
            'ex_accuracy': sum(grades) / len(grades),
            'lf_accuracy': sum(exact_match) / len(exact_match),
            }
        return result


def eval_all(incremental=True):
    for expname in glob.iglob("*"):
        if re.match("wikisql\_s2s.*\_clean.*", expname):
            settings = json.load(open(os.path.join("", expname, "settings.json")))
            if settings["completed"] == True and settings["epochs"] == 50:
                if os.path.is_file(os.path.join("", expname, "exec_eval_results.json")):
                    if incremental:
                        continue
                try:
                    print(expname)
                    # get dev numbers
                    pred_file = os.path.join("", expname, "dev_pred.jsonl")
                    source_file = "../../../datasets/wikisql_clean/dev.jsonl"
                    db_file = "../../../datasets/wikisql_clean/dev.db"
                    dev_results = eval_one_qelos(db_file, pred_file, source_file)
                    # get test numbers
                    pred_file = os.path.join("", expname, "test_pred.jsonl")
                    source_file = "../../../datasets/wikisql_clean/test.jsonl"
                    db_file = "../../../datasets/wikisql_clean/test.db"
                    test_results = eval_one_qelos(db_file, pred_file, source_file)
                    # write out
                    out_results = {"dev_exec_acc": dev_results["ex_accuracy"],
                                   "dev_seq_acc": dev_results["lf_accuracy"],
                                   "test_exec_acc": test_results["ex_accuracy"],
                                   "test_seq_acc": test_results["lf_accuracy"]}
                    print(json.dumps(out_results, indent=3, sort_keys=True))
                    with open(os.path.join("", expname, "exec_eval_results.json"), "w") as f:
                        json.dump(out_results, f)
                except Exception as e:
                    print("{} failed".format(expname))


DATA_PATH = "../../../datasets/wikisql_clean/"


def get_accuracies(p, verbose=False):
    if verbose:
        print(p)
    # dev numbers
    pred_file = os.path.join(p, "dev_pred.jsonl")
    source_file = os.path.join(DATA_PATH, "dev.jsonl")
    db_file = os.path.join(DATA_PATH, "dev.db")
    dev_results = eval_one_qelos(db_file, pred_file, source_file)
    # pred numbers
    pred_file = os.path.join(p, "test_pred.jsonl")
    source_file = os.path.join(DATA_PATH, "test.jsonl")
    db_file = os.path.join(DATA_PATH, "test.db")
    test_results = eval_one_qelos(db_file, pred_file, source_file)

    # write out
    out_results = {"dev_exec_acc": dev_results["ex_accuracy"],
                   "dev_seq_acc": dev_results["lf_accuracy"],
                   "test_exec_acc": test_results["ex_accuracy"],
                   "test_seq_acc": test_results["lf_accuracy"]}
    if verbose:
        print(json.dumps(out_results, indent=3, sort_keys=True))
    with open(os.path.join("", p, "exec_eval_results.json"), "w") as f:
        json.dump(out_results, f)

    return dev_results["lf_accuracy"], dev_results["ex_accuracy"], test_results["lf_accuracy"], test_results["ex_accuracy"]


def get_avg_accs_of(*args, **kw):
    """ signature is forward to find_experiments(*args, **kw) to find matching experiments
        get_accuracies() is run for every found experiment and the average is returned """
    experiment_dirs = list(find_experiments(*args, **kw))
    accses = [[] for i in range(2)]
    for experiment_dir in experiment_dirs:
        accs = get_accuracies(experiment_dir, verbose=True)
        for acc, accse in zip(accs, accses):
            accse.append(acc * 100.)
    print("Average accs for {} selected experiments:".format(len(accses[0])))
    print("  DEV LF ACC: {:.2f}, std={:.2f}".format(np.mean(accses[0]), np.std(accses[0])))
    print("  DEV EXE ACC: {:.2f}, std={:.2f}".format(np.mean(accses[1]), np.std(accses[1])))
    print("  TEST LF ACC: {:.2f}, std={:.2f}".format(np.mean(accses[2]), np.std(accses[2])))
    print("  TEST EXE ACC: {:.2f}, std={:.2f}".format(np.mean(accses[3]), np.std(accses[3])))

    return accses


def find_experiments(*args, **kw):
    """ finds directories satisfying settings conditions in kw (as recorded by logger)
        if any, the first element of *args will always be interpreted as a prefix to filter subdirs by,
            and the second element of *args will be interpreted as an alternative path to search experiments in
    """
    p = None if len(args) < 2 else args[1]
    prefix = None if len(args) < 1 else args[0]
    if p is None:
        p = "."
    for subdir, dirs, files in os.walk(p):
        if "settings.json" in files:
            settings = json.load(open(os.path.join(subdir, "settings.json")))
            incl = True
            if prefix is not None:
                incl &= re.match(prefix, subdir) is not None
            for k, v in kw.items():
                if k not in settings:
                    settings[k] = None
                if iscallable(v):
                    incl &= v(settings[k])
                else:
                    incl &= settings[k] == v
                if not incl:
                    break
            if incl:
                yield subdir


def iscallable(x):
    return hasattr(x, "__call__")


if __name__ == "__main__":
    eval_all()
