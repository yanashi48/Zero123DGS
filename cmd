#生成多视角图
python generate2.py --ckpt ckpt/syncdreamer-pretrain.ckpt \
                   --input testset/poro.png \
                   --output output/poro/images \
                   --sample_num 1 \
                   --cfg_scale 2.0 \
                   --elevation 30 \
                   --crop_size 200
                

#colmap生成cameras.txt images.txt
python eval_colmap.py --dir output/poro/images --project output/poro --name poro --colmap /usr/local/bin/colmap

#生成masks
python amg.py --checkpoint models/sam_vit_h_4b8939.pth --model-type default --input output/poro/images --output output/poro/masks

#生成visual hull 
python visual_hull.py \
    --sparse_id 16 \
    --data_dir output/poro \
    --reso 1 --not_vis

#生成3DGS
python train_gs.py -s output/poro \
    -m output/poro/output/gs_init/poro \
    -r 1 --sparse_view_num 16 --sh_degree 2 \
    --init_pcd_name visual_hull_16 \
    --white_background --random_background

#测试多视角图像与3DGS渲染结果的指标  
python render.py \
    -m output/poro/output/gs_init/poro \
    --sparse_view_num 16 --sh_degree 2 \
    --init_pcd_name visual_hull_16 \
    --white_background --skip_all --skip_train

#360度渲染3DGS结果   
python render.py \
    -m output/poro/output/gs_init/poro \
    --sparse_view_num 16 --sh_degree 2 \
    --init_pcd_name visual_hull_16 \
    --white_background --render_path

