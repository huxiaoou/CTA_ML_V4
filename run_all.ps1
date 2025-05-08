Remove-Item E:\Data\Projects\CTA_ML_V4\* -Recurse

$bgn_date_avlb = "20120104"
$bgn_date_factor = "20140102"
$bgn_date_mclrn = "20150105"
$bgn_date_sig = "20151230" # must at least 2 days ahead of bgn date
$bgn_date = "20160104"
$stp_date = "20250401"

python main.py --bgn $bgn_date_avlb --stp $stp_date available
python main.py --bgn $bgn_date_avlb --stp $stp_date market
python main.py --bgn $bgn_date_avlb --stp $stp_date test_return
python main.py --bgn $bgn_date_factor --stp $stp_date factor --fclass MTM
python main.py --bgn $bgn_date_factor --stp $stp_date factor --fclass SKEW
python main.py --bgn $bgn_date_factor --stp $stp_date factor --fclass KURT
python main.py --bgn $bgn_date_factor --stp $stp_date factor --fclass RS
python main.py --bgn $bgn_date_factor --stp $stp_date factor --fclass BASIS
python main.py --bgn $bgn_date_factor --stp $stp_date factor --fclass TS
python main.py --bgn $bgn_date_factor --stp $stp_date factor --fclass LIQUIDITY
python main.py --bgn $bgn_date_factor --stp $stp_date factor --fclass SIZE

python main.py --bgn $bgn_date --stp $stp_date ic --fclass MTM
python main.py --bgn $bgn_date --stp $stp_date ic --fclass SKEW
python main.py --bgn $bgn_date --stp $stp_date ic --fclass KURT
python main.py --bgn $bgn_date --stp $stp_date ic --fclass RS
python main.py --bgn $bgn_date --stp $stp_date ic --fclass BASIS
python main.py --bgn $bgn_date --stp $stp_date ic --fclass TS
python main.py --bgn $bgn_date --stp $stp_date ic --fclass LIQUIDITY
python main.py --bgn $bgn_date --stp $stp_date ic --fclass SIZE

python main.py --bgn $bgn_date_mclrn --stp $stp_date mclrn
python main.py --bgn $bgn_date_sig --stp $stp_date signals

python main.py --bgn $bgn_date --stp $stp_date quick
python main.py --bgn $bgn_date --stp $stp_date --nomp simulations
python main.py --bgn $bgn_date --stp $stp_date evaluations
