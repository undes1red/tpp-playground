import numpy as np

from src.toolbox.misc import dump_to_pkl, get_logger, write_to_txt

logger = get_logger(name=__file__)


def get_explanation_postprocess(all_evaluation_results, desc, opt):
    """
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    """
    (
        the_number_of_remained_events,
        picked_perplexity_p_h_d_x_o,
        picked_perplexity_gap_between_distilled_and_full,
        picked_perplexity_p_h_l_x_o,
        picked_perplexity_gap_between_left_and_full,
    ) = all_evaluation_results

    avg_the_number_of_remained_events = np.mean(the_number_of_remained_events)
    avg_picked_perplexity_p_h_d_x_o = np.mean(picked_perplexity_p_h_d_x_o)
    avg_picked_perplexity_gap_between_distilled_and_full = np.mean(picked_perplexity_gap_between_distilled_and_full)
    avg_picked_perplexity_p_h_l_x_o = np.mean(picked_perplexity_p_h_l_x_o)
    avg_picked_perplexity_gap_between_left_and_full = np.mean(picked_perplexity_gap_between_left_and_full)

    result_file = opt.store_dir / f"{desc}_get_explanations.txt"
    strings = [f"For the {desc} of {opt.dataset_name},\n",
               f"The average length of the H_d is {avg_the_number_of_remained_events}.\n",
               f"The average of ppl p(x_o|h_d) is {avg_picked_perplexity_p_h_d_x_o}.\n",
               f"The average of ppl p(x_o|h_l) is {avg_picked_perplexity_p_h_l_x_o}.\n",
               f"The average gap between epsilon_d and of ppl p(x_o|h_d) is {avg_picked_perplexity_gap_between_distilled_and_full}.\n",
               f"The average gap between epsilon_l and of ppl p(x_o|h_l) is {avg_picked_perplexity_gap_between_left_and_full}.\n",
               ]
    write_to_txt(strings, result_file)

    result_dist_file = opt.store_dir / f"{desc}_get_explanation.pkl"
    dump_to_pkl(
        {
            "length": the_number_of_remained_events,
            "perplexity_p_h_d_x_o": picked_perplexity_p_h_d_x_o,
            "perplexity_p_h_l_x_o": picked_perplexity_p_h_l_x_o,
            "perplexity_gap_between_distilled_and_full": picked_perplexity_gap_between_distilled_and_full,
            "perplexity_gap_between_left_and_full": picked_perplexity_gap_between_left_and_full,

        }, result_dist_file, compression="bz2")

desc_funcs = {
    "get_explanation": {
        "desc_string": "Generating explanations for {0}",
        "postprocess_func": get_explanation_postprocess,
    },
}
