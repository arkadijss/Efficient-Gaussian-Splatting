bash runsh/apply_light_prune.sh output/orig/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/orig/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/orig/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/orig/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/orig/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/orig/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/orig/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/orig/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/orig/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/orig/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/orig/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/orig/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/orig/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/orig/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/compact3d/bicycle 360_v2/bicycle true 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/compact3d/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/compact3d/bicycle/point_cloud/iteration_30000/point_cloud_decompressed.ply 
python mini_splatting_compress/render.py -m output/compact3d/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/compact3d/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/compact3d/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/compact3d/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/compact3d/bicycle 360_v2/bicycle true 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/compact3d/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/compact3d/bicycle/point_cloud/iteration_30000/point_cloud_decompressed.ply 
python mini_splatting_compress/render.py -m output/compact3d/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/compact3d/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/compact3d/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/compact3d/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_6/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_opacity_6/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_6/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_6/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_6/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_6/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_6/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_6/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_opacity_6/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_6/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_6/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_6/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_6/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_6/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_5/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_opacity_5/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_5/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_5/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_5/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_5/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_5/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_5/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_opacity_5/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_5/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_5/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_5/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_5/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_5/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/compact3d_noscale/bicycle 360_v2/bicycle true 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/compact3d_noscale/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/compact3d_noscale/bicycle/point_cloud/iteration_30000/point_cloud_decompressed.ply 
python mini_splatting_compress/render.py -m output/compact3d_noscale/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/compact3d_noscale/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/compact3d_noscale/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/compact3d_noscale/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/compact3d_noscale/bicycle 360_v2/bicycle true 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/compact3d_noscale/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/compact3d_noscale/bicycle/point_cloud/iteration_30000/point_cloud_decompressed.ply 
python mini_splatting_compress/render.py -m output/compact3d_noscale/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/compact3d_noscale/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/compact3d_noscale/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/compact3d_noscale/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_8/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_8/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_8/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_8/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_8/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_8/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_8/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_8/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_8/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_8/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_8/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_8/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_8/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_8/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_2/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_opacity_2/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_2/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_2/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_2/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_2/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_2/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_2/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_opacity_2/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_2/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_2/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_2/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_2/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_2/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_3/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_3/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_3/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_3/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_3/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_3/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_3/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_3/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_3/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_3/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_3/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_3/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_3/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_3/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/orig_depth_reg/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/orig_depth_reg/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/orig_depth_reg/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/orig_depth_reg/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/orig_depth_reg/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/orig_depth_reg/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/orig_depth_reg/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/orig_depth_reg/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/orig_depth_reg/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/orig_depth_reg/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/orig_depth_reg/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/orig_depth_reg/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/orig_depth_reg/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/orig_depth_reg/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_1/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_1/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_1/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_1/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_1/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_1/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_1/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_1/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_1/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_1/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_1/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_1/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_1/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_1/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_7/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_7/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_7/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_7/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_7/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_7/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_7/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_7/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_7/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_7/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_7/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_7/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_7/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_7/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/orig_opacity/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/orig_opacity/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/orig_opacity/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/orig_opacity/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/orig_opacity/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/orig_opacity/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/orig_opacity/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/orig_opacity/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/orig_opacity/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/orig_opacity/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/orig_opacity/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/orig_opacity/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/orig_opacity/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/orig_opacity/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_1/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_opacity_1/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_1/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_1/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_1/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_1/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_1/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_1/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_opacity_1/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_1/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_1/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_1/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_1/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_1/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_5/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_5/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_5/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_5/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_5/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_5/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_5/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_5/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_5/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_5/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_5/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_5/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_5/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_5/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_compact3d/bicycle 360_v2/bicycle true 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_compact3d/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_compact3d/bicycle/point_cloud/iteration_30000/point_cloud_decompressed.ply 
python mini_splatting_compress/render.py -m output/mini_compact3d/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_compact3d/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_compact3d/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_compact3d/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_compact3d/bicycle 360_v2/bicycle true 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_compact3d/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_compact3d/bicycle/point_cloud/iteration_30000/point_cloud_decompressed.ply 
python mini_splatting_compress/render.py -m output/mini_compact3d/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_compact3d/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_compact3d/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_compact3d/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_7/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_opacity_7/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_7/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_7/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_7/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_7/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_7/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity_7/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_opacity_7/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_opacity_7/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity_7/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity_7/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity_7/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity_7/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity/bicycle 360_v2/bicycle false 0.66 1 0.2 
bash runsh/apply_light_distill.sh output/mini_opacity/light_prune_0.66_1_0.2_v_important_score/bicycle 360_v2/bicycle output/mini_opacity/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity/light_prune_0.66_1_0.2_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity/light_prune_0.66_1_0.2_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity/light_prune_0.66_1_0.2_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

bash runsh/apply_light_prune.sh output/mini_opacity/bicycle 360_v2/bicycle false 0.8 1 0.1 
bash runsh/apply_light_distill.sh output/mini_opacity/light_prune_0.8_1_0.1_v_important_score/bicycle 360_v2/bicycle output/mini_opacity/bicycle/point_cloud/iteration_30000/point_cloud.ply 
python mini_splatting_compress/render.py -m output/mini_opacity/mini_s/bicycle -s 360_v2/bicycle --skip_train --overwrite 
bash runsh/apply_light_vq.sh output/mini_opacity/light_prune_0.8_1_0.1_v_important_score_distill/bicycle 360_v2/bicycle true 
python light_gauss/render.py -m output/mini_opacity/light_prune_0.8_1_0.1_v_important_score/bicycle -s 360_v2/bicycle --skip_train --overwrite --iteration 5000 
python light_gauss/render.py -m output/mini_opacity/light_prune_0.8_1_0.1_v_important_score_distill_vq/bicycle -s 360_v2/bicycle --skip_train --overwrite --load_vq --iteration "-10" 

