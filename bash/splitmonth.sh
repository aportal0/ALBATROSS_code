for year in 1996; do
    year="${year%/}"
    echo $year
    cd "$year"
    cdo splitmon "t2m_daily_maximum_${year}.nc" "t2m_daily_maximum_try_${year}"
    cd ..
done

