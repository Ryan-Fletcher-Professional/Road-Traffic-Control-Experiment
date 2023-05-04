import pandas as pd
import scipy.stats as stats

# The alpha value we consider as threshold
ALPHA = 0.05

# Check whether a Wilcoxon signed rank test rejected or accpeted the null hypothesis
def significanceCheck(pvalue):
    if pvalue < ALPHA:
        return "Reject null hypothesis."
    else:
        return "Accept null hypothesis."

# Conduct a Wilcoxon signed rank test with a two-sided alternative hypothesis
def runWilcoxonSignedRankTest(col1, col2, csv_file1, csv_file2, csv_file3, out_file):
    print(f"Wilcoxon signed rank test on {col1} and {col2} for system mean speed", file=out_file)
    wilcoxon_result = stats.wilcoxon(csv_file1[col1], csv_file1[col2], alternative='less')
    print(significanceCheck(wilcoxon_result.pvalue), file=out_file)
    
    print(f"Wilcoxon signed rank test on {col1} and {col2} for system total waiting time", file=out_file)
    wilcoxon_result = stats.wilcoxon(csv_file2[col1], csv_file2[col2], alternative='greater')
    print(significanceCheck(wilcoxon_result.pvalue), file=out_file)
    
    print(f"Wilcoxon signed rank test on {col1} and {col2} for CO2 emissions", file=out_file)
    wilcoxon_result = stats.wilcoxon(csv_file3[col1], csv_file3[col2], alternative='greater')
    print(significanceCheck(wilcoxon_result.pvalue), file=out_file)

data_sys_mean_speed = pd.read_csv("milestone2/sys_mean_speed.csv")
data_sys_total_waiting_time = pd.read_csv("milestone2/sys_total_waiting_time.csv")
data_co2_emissions = pd.read_csv("milestone2/co2_emissions_mg.csv")

list_of_col_names = data_sys_mean_speed.columns;

with open(r"milestone2\wilcoxon_signed_rank_test.results", 'w') as out_file:
    runWilcoxonSignedRankTest('FTSC', list_of_col_names[2], data_sys_mean_speed, data_sys_total_waiting_time, data_co2_emissions, out_file)

    runWilcoxonSignedRankTest('FTSC', list_of_col_names[3], data_sys_mean_speed, data_sys_total_waiting_time, data_co2_emissions, out_file)

    runWilcoxonSignedRankTest('FTSC', list_of_col_names[4], data_sys_mean_speed, data_sys_total_waiting_time, data_co2_emissions, out_file)

    runWilcoxonSignedRankTest('FTSC', list_of_col_names[5], data_sys_mean_speed, data_sys_total_waiting_time, data_co2_emissions, out_file)

    runWilcoxonSignedRankTest(list_of_col_names[2], list_of_col_names[3], data_sys_mean_speed, data_sys_total_waiting_time, data_co2_emissions, out_file)

    runWilcoxonSignedRankTest(list_of_col_names[4], list_of_col_names[5], data_sys_mean_speed, data_sys_total_waiting_time, data_co2_emissions, out_file)