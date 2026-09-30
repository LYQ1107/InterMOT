# N72R20 Storage Audit Before

- Timestamp (UTC): `2026-09-30T07:08:05.632970+00:00`
- `/data3/liuyeqiang` filesystem: `/dev/loop10    1007G  770G  187G  81% /data3/liuyeqiang`
- Download performed: `false`
- Cleanup performed: `false`

## Mounted filesystems

```text
Filesystem      Size  Used Avail Use% Mounted on
tmpfs            13G  4.0M   13G   1% /run
/dev/nvme0n1p2  3.6T   45G  3.4T   2% /
tmpfs            63G  251M   63G   1% /dev/shm
tmpfs           5.0M     0  5.0M   0% /run/lock
efivarfs        512K   51K  457K  10% /sys/firmware/efi/efivars
/dev/nvme0n1p1  1.1G  6.2M  1.1G   1% /boot/efi
/dev/nvme1n1    3.6T   71G  3.4T   3% /data1
/dev/nvme2n1    3.6T  1.2T  2.3T  33% /data2
/dev/nvme3n1    3.6T  1.4T  2.1T  40% /data3
tmpfs            13G   96K   13G   1% /run/user/120
/dev/loop9      2.0T   70G  1.8T   4% /data1/guoruohao
/dev/loop10    1007G  770G  187G  81% /data3/liuyeqiang
tmpfs            13G   84K   13G   1% /run/user/1001
/dev/loop11    1007G  483G  474G  51% /data3/weiyijie
/dev/loop12    1007G  127G  829G  14% /data2/geshuyu
/dev/loop13    1007G  547G  410G  58% /data2/dongchenwei
/dev/loop14     1.5T  414G 1021G  29% /data2/yejiatong
tmpfs            13G   84K   13G   1% /run/user/1006
tmpfs            13G   88K   13G   1% /run/user/1005
tmpfs            13G   84K   13G   1% /run/user/1002
/dev/loop20    1007G   63G  893G   7% /data3/liweiran
tmpfs            13G   88K   13G   1% /run/user/1004
tmpfs            13G   84K   13G   1% /run/user/1003
```

## Sized paths and files (descending)

| GiB | Exists | Label | Path |
|---:|:---:|---|---|
| 18.4306 | yes | InterMOT repository | `/data3/liuyeqiang/InterMOT` |
| 11.6768 | yes | N72R16 assets | `/data3/liuyeqiang/InterMOT_N72R16_assets` |
| 10.5033 | yes | DanceTrack candidate root | `/data3/liuyeqiang/InterMOT_N72R16_assets/dataset` |
| 9.8439 | yes | N72R19R1 outputs | `/data3/liuyeqiang/InterMOT/outputs/N72R19R1` |
| 6.5087 | yes | DanceTrack train | `/data3/liuyeqiang/InterMOT_N72R16_assets/dataset/train` |
| 3.9946 | yes | DanceTrack val | `/data3/liuyeqiang/InterMOT_N72R16_assets/dataset/val` |
| 2.4203 | yes | pip cache | `/data3/liuyeqiang/.cache/pip` |
| 1.5990 | yes | N72R17 assets | `/data3/liuyeqiang/InterMOT_N72R17_assets` |
| 1.3564 | yes | N72R19 outputs | `/data3/liuyeqiang/InterMOT/outputs/N72R19` |
| 0.5637 | yes | HuggingFace cache | `/data3/liuyeqiang/.cache/huggingface` |
| 0.3252 | yes | research_references | `/data3/liuyeqiang/research_references` |
| 0.1544 | yes | Torch cache | `/data3/liuyeqiang/.cache/torch` |
| 0.0659 | yes | N72R18 outputs | `/data3/liuyeqiang/InterMOT/outputs/N72R18` |
| 0.0001 | yes | Codex temporary directory | `/data3/liuyeqiang/.codex/.tmp` |

No cleanup is authorized by this audit. Any later cleanup must first appear in `cleanup_candidates.json` and `storage_cleanup_log.jsonl`.
