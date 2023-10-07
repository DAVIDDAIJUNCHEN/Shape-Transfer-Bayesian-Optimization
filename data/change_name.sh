#!/usr/bin/env bash 

for i in $(seq 1 20); do 
	mv $i/simExp_points_task2_bcbo.tsv $i/simExp_points_task2_bcbo_from_gp.tsv
	mv $i/simExp_points_task2_gp.tsv   $i/simExp_points_task2_gp_from_gp.tsv
	mv $i/simExp_points_task2_stbo.tsv $i/simExp_points_task2_stbo_from_gp.tsv
done 

