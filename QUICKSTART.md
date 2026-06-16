# Looped MoE 



# ===================================================================    
# Regular size configs  

## DEEPSEEK
# g1
python csub.py -n g1-ds -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --tie-group-size 1 --grad-accum 16 --batch-size 4 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g1-ds.log"
    
# g4
python csub.py -n g4-ds -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --tie-group-size 4 --grad-accum 8 --batch-size 8 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g4-ds.log"
    
# g1 narrow width shrinking 4x for iso-param  (16 = 64/4), notie
python csub.py -n g1-ds-narrow -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --tie-group-size 1 --expand-tied-experts 16 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g1-ds-narrow.log"  
    
# width expansion 2x
python csub.py -n g4-ds-w2x -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --tie-group-size 4 --expand-tied-experts 128 --grad-accum 8 --batch-size 8 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g4-ds-w2x.log"      

# width expansion 4x
python csub.py -n g4-ds-w4x -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --tie-group-size 4 --expand-tied-experts 256 --grad-accum 8 --batch-size 8 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g4-ds-w4x.log"

    
## QWEN
# g1
python csub.py -n g1-qw -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --tie-group-size 1 --grad-accum 16 --batch-size 4 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g1-qw.log"
   
# g4
python csub.py -n g4-qw -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --tie-group-size 4 --grad-accum 8 --batch-size 8 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g4-qw.log"
    
# g1 narrow width shrinking 4x for iso-param  (15 = 60/4), notie
python csub.py -n g1-qw-wd4 -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --tie-group-size 1 --expand-tied-experts 15 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g1-qw-wd4.log"  

# width expansion 2x
python csub.py -n g4-qw-w2x -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --tie-group-size 4 --expand-tied-experts 120 --grad-accum 8 --batch-size 8 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g4-qw-w2x.log"      

# 4x
python csub.py -n g4-qw-w4x -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --tie-group-size 4 --expand-tied-experts 240 --grad-accum 8 --batch-size 8 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume  2>&1 | tee -a logs/g4-qw-w4x.log"
    
    
    
## OLMOE

# g1
python csub.py -n g1-olm -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --tie-group-size 1 --grad-accum 16 --batch-size 4 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume 2>&1 | tee -a logs/g1-olm.log"

# g4
python csub.py -n g4-olm -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --tie-group-size 4 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume 2>&1 | tee -a logs/g4-olm.log"
    
# width expansion 2x
python csub.py -n g4-olm-w2x -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --tie-group-size 4 --expand-tied-experts 128 --grad-accum 8 --batch-size 8 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume 2>&1 | tee -a logs/g4-olm-w2x.log"    

# 4x
python csub.py -n g4-olm-w4x -g 4 --node-type h200 --train -t 80h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --tie-group-size 4 --expand-tied-experts 256 --grad-accum 8 --batch-size 8 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --n-steps 30000 --auto-resume 2>&1 | tee -a logs/g4-olm-w4x.log"






# ===================================================================    
# Tiny size configs  
    
## tiny DEEPSEEK    (seq len 2k)
# g1
python csub.py -n tiny-g1-ds -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch deepseek --scale tiny --tie-group-size 1 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g1-ds.log"

# g1 width shrinking 4x for iso-param  (16 = 64/4), notie
python csub.py -n tiny-g1-ds-narrow -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch deepseek --scale tiny --tie-group-size 1 --expand-tied-experts 16 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g1-ds-narrow.log"  
    
# g4
python csub.py -n tiny-g4-ds -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch deepseek --scale tiny --tie-group-size 4 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-ds.log"
    
# width expansion 2x
python csub.py -n tiny-g4-ds-w2x -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch deepseek --scale tiny --tie-group-size 4 --expand-tied-experts 128 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-ds-w2x.log"      

# width expansion 4x
python csub.py -n tiny-g4-ds-w4x -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch deepseek --scale tiny --tie-group-size 4 --expand-tied-experts 256 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-ds-w4x.log"
    
    
    
## tiny QWEN
# g1
python csub.py -n tiny-g1-qw -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch qwen3moe --scale tiny --tie-group-size 1 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g1-qw.log"
   
# g1 width shrinking 4x for iso-param  (15 = 60/4), notie
python csub.py -n tiny-g1-qw-narrow -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch qwen3moe --scale tiny --tie-group-size 1 --expand-tied-experts 15 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g1-qw-narrow.log" 
    
# g4
python csub.py -n tiny-g4-qw -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch qwen3moe --scale tiny --tie-group-size 4 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-qw.log"
    
# width expansion 2x
python csub.py -n tiny-g4-qw-w2x -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch qwen3moe --scale tiny --tie-group-size 4 --expand-tied-experts 120 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-qw-w2x.log"      

# 4x
python csub.py -n tiny-g4-qw-w4x -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch qwen3moe --scale tiny --tie-group-size 4 --expand-tied-experts 240 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-qw-w4x.log"
    
    
    
## tiny OLMOE
# g1
python csub.py -n tiny-g1-olm -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch olmoe --scale tiny --tie-group-size 1 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g1-olm.log"
   
# g1 width shrinking 4x for iso-param  (16 = 64/4), notie
python csub.py -n tiny-g1-olm-narrow -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch olmoe --scale tiny --tie-group-size 1 --expand-tied-experts 16 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g1-olm-narrow.log" 
   
# g4 
python csub.py -n tiny-g4-olm -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch olmoe --scale tiny --tie-group-size 4 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-olm.log"
    
# width expansion 2x
python csub.py -n tiny-g4-olm-w2x -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch olmoe --scale tiny --tie-group-size 4 --expand-tied-experts 128 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-olm-w2x.log"      

# 4x
python csub.py -n tiny-g4-olm-w4x -g 1 --node-type h200 --train -t 28h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python moe_train.py --arch olmoe --scale tiny --tie-group-size 4 --expand-tied-experts 256 --grad-accum 16 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-g4-olm-w4x.log"


  
______________________________
# DDP4 tiny runs (20k steps)

# ===================================================================   
# TINY   
# ===================================================================   
    
## tiny DEEPSEEK    (seq len 2k)
# g1
python csub.py -n tiny-g1-ds -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --scale tiny --tie-group-size 1 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g1-ds.log"
    
# g4
python csub.py -n tiny-g4-ds -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --scale tiny --tie-group-size 4 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-ds.log"
    
# g2 
python csub.py -n tiny-g2-ds -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --scale tiny --tie-group-size 2 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 1.41 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume 2>&1 | tee -a logs/tiny-ddp-g2-ds.log"
    
# width expansion 2x
python csub.py -n tiny-g4-ds-w2x -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --scale tiny --tie-group-size 4 --expand-tied-experts 128 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-ds-w2x.log"
    
# width expansion 4x
python csub.py -n tiny-g4-ds-w4x -g 4 --node-type h200 --train -t 18h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --scale tiny --tie-group-size 4 --expand-tied-experts 256 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-ds-w4x.log"
    
    
    
## tiny QWEN
# g1
python csub.py -n tiny-g1-qw -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --scale tiny --tie-group-size 1 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g1-qw.log"
    
# g4
python csub.py -n tiny-g4-qw -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --scale tiny --tie-group-size 4 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-qw.log"
    
# g2 
python csub.py -n tiny-g2-qw -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --scale tiny --tie-group-size 2 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 1.41 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume 2>&1 | tee -a logs/tiny-ddp-g2-qw.log"
    
# width expansion 2x
python csub.py -n tiny-g4-qw-w2x -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --scale tiny --tie-group-size 4 --expand-tied-experts 120 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-qw-w2x.log"
    
# width expansion 4x
python csub.py -n tiny-g4-qw-w4x -g 4 --node-type h200 --train -t 18h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch qwen3moe --scale tiny --tie-group-size 4 --expand-tied-experts 240 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-qw-w4x.log"
    
    
    
## tiny OLMOE
# g1
python csub.py -n tiny-g1-olm -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --scale tiny --tie-group-size 1 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 1.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g1-olm.log"
   
# g4
python csub.py -n tiny-g4-olm -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --scale tiny --tie-group-size 4 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-olm.log"
    
# g2 
python csub.py -n tiny-g2-olm -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --scale tiny --tie-group-size 2 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 1.41 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume 2>&1 | tee -a logs/tiny-ddp-g2-olm.log"
    
# width expansion 2x
python csub.py -n tiny-g4-olm-w2x -g 4 --node-type h200 --train -t 12h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --scale tiny --tie-group-size 4 --expand-tied-experts 128 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-olm-w2x.log"
    
# width expansion 4x
python csub.py -n tiny-g4-olm-w4x -g 4 --node-type h200 --train -t 18h \
    --command "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && torchrun --standalone --nproc_per_node=4 moe_train.py --arch olmoe --scale tiny --tie-group-size 4 --expand-tied-experts 256 --grad-accum 4 --batch-size 16 --optimizer muon --tied-lr-divisor 2.0 --z-loss-coef 1e-4 --no-gradient-checkpointing --n-steps 20000 --auto-resume  2>&1 | tee -a logs/tiny-ddp-g4-olm-w4x.log"
