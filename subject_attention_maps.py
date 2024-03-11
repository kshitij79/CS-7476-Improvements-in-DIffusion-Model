import numpy as np

def extract_subject_attention_maps(attention_maps, subject_info):
    subject_attention_maps = {}

    for subject, idx in subject_info.items():
        subject_attention_maps[subject] = {}
        for attention_key in attention_maps.keys():
            attention_map = attention_maps[attention_key]
            subject_attention_map = attention_map[:, :, :, idx]
            # only keep 0th batch as it captues conditional info
            subject_attention_map = subject_attention_map[0]
            subject_attention_maps[subject][attention_key] = subject_attention_map

    return subject_attention_maps