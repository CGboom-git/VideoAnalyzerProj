### 视频行为分析系统v2

* 作者：北小菜 
* 官网：http://www.beixiaocai.com
* 邮箱：bilibili_bxc@126.com
* QQ：1402990689
* 微信：bilibili_bxc
* 哔哩哔哩主页：https://space.bilibili.com/487906612
* gitee开源地址：https://gitee.com/Vanishi/BXC_VideoAnalyzer_v2
* github开源地址：https://github.com/any12345com/BXC_VideoAnalyzer_v2

### 软件版本介绍
* v1版本开源地址 https://gitee.com/Vanishi/BXC_VideoAnalyzer_v1
* v2版本开源地址 https://gitee.com/Vanishi/BXC_VideoAnalyzer_v2
* v3版本安装包下载地址 https://gitee.com/Vanishi/BXC_VideoAnalyzer_v3
* v4版本安装包下载地址 https://gitee.com/Vanishi/BXC_VideoAnalyzer_v4

### v2版本架构介绍
~~~
（1）Analyzer_v2:  基于C++开发的视频分析器模块,主要实现视频流的推拉流和编解码
（2）Admin_v2:     基于Python开发的后台管理模块
（3）Algorithm_v2: 基于Python开发的算法模块，主要实现了yolo5和ssd的目标检测算法，使用flask对外提供算法接口服务，供分析器模块调用
（4）zlm:       基于C++开发的流媒体模块。也可以使用我编译的其他版本：https://gitee.com/Vanishi/zlm				

注意：Analyzer_v2是兼容linux系统的。
	  在编译Analyzer时需要的依赖库源码和编译文档我放在了网盘中，网盘链接：https://pan.quark.cn/s/9cf9832d6e8a 提取码：x34N
	  Admin和Algorithm都是基于Python开发，自身是跨平台。
	  zlm也是支持跨平台的，但我仅编译了windows版本zlm，大家需要linux版本zlm，需要自行编译。ZLMediaKit开源地址：https://gitee.com/xia-chu/ZLMediaKit

~~~

### v2版本相比于v1版本的变化

~~~
（1）Anaylzer_v2 
一，在支持Windows的基础上兼容Linux （如果有需要部署arm或者其他嵌入式平台，也可以尝试编译）
二，删除了算法检测模块动态库和和合成报警视频模块动态库，将动态库的代码简化后提取出来，与主程序代码放在一个项目中，提高了代码整体可读性，并优化了合成报警视频模块的帧复用
三，调用算法模块删除了C++调用Python版，仅保留了C++调用Python接口版的方式，简化了代码，提高了代码可读性，非常有利于学习和理解，此前因为代码要兼容C++调用Python的方式，所以很多朋友在学习这部分的时候，不太容易看明白，现在就清晰多了。
四，算法渲染后流推流由rtmp推流改成了rtsp推流，并修复了此前sps，pps的设置问题。
五，考虑到这个项目目前并没有对音频流做算法分析，因此本次代码升级，删除了音频的解码和编码模块，后续大家如有需要对音频做编解码处理，可以参考之前的v1版本，或者参考我的开源远程控制软件SRE中有关音频编解码的使用，SRE中的音频编解码相关的操作，更加的完善，更加的具有借鉴参考价值。
 		SRE远程控制软件开源地址：https://gitee.com/Vanishi/SRE

（2）Admin_v2
一，强调一点，因为基于Python开发的，所以此前的v1版本和当前的v2版本都是支持跨平台的。
二，去除了一些无效的功能，修复了一些存在bug的功能
三，优化了错误提示，新版本的错误提示，可以很明显的看出来，是哪一个模块出了问题，程序启动后，可以在操作过程中，看到具体哪个模块报错。

（3）Algorithm_v2 
相比于v1，无变化

（4）zlm （基于ZLMediaKit编译的流媒体服务器）
相比于v1，无变化。大家也可以自行下载ZLMediaKit源码进行编译使用。ZLMediaKit开源地址：https://gitee.com/xia-chu/ZLMediaKit

~~~



### 相关视频链接
* v1版本视频介绍地址 https://www.bilibili.com/video/BV1dG4y1k77o
* v1版本源码讲解（1）拉流，解码，实时算法分析，合成报警视频，编码，推流 https://www.bilibili.com/video/BV1L84y177xc
* v1版本源码讲解（2）音频解码，音频重采样，音频编码，合成报警视频 https://www.bilibili.com/video/BV1984y1L7zB
* v2版本视频介绍地址 https://www.bilibili.com/video/BV1CG411f7ak
* v3版本视频介绍地址 https://www.bilibili.com/video/BV1Xy4y1P7M2

### ffmpeg命令行推流模拟摄像头

~~~

//将本地文件推流至VideoAnalyzer（该命令行未经优化，延迟较大）
ffmpeg -re -stream_loop -1  -i test.mp4  -rtsp_transport tcp -c copy -f rtsp rtsp://127.0.0.1:554/live/test

//将摄像头视频流推流至VideoAnalyzer（该命令行已优化，但仍然存在延迟，如果想要彻底解决推流延迟，可以参考我的视频：https://space.bilibili.com/487906612）
ffmpeg  -rtsp_transport tcp -i "视频源地址" -fflags nobuffer -max_delay 1 -threads 5  -profile:v high  -preset superfast -tune zerolatency  -an -c:v h264 -crf 25 -s 1280*720   -f rtsp -bf 0  -g 5  -rtsp_transport tcp rtsp://127.0.0.1:554/live/camera

// 备注
根目录下data文件夹中，我提供了一个test.mp4，大家可以测试，模拟视频流

~~~

### 有关ffmpeg推流的几点补充说明

* 通过ffmpeg命令行实现的推流功能，延迟总是存在的，且无法解决。但基于ffmpeg开发库却可以彻底解决延迟推流的问题，可以参考我的视频：https://space.bilibili.com/487906612


### 这套软件适合哪些人？

~~~
（1）如果是做智慧安防的公司，拿来之后稍微完善包装一下，扩展一下算法功能就能投入实用。
（2）如果是做相关创业的朋友，拿来之后在此基础上二次开发一些功能即可，可以省下一些开发岗位的工作，节省成本。
（3）如果是做算法的朋友，可以套用整个软件的运行流程，只需要扩展自己的算法，就能在此基础上做出比较实用的产品，也可以让自己做的算法快速的形成产品，快速的体验算法的效果。
（4）如果是做算法或软件相关毕业论文的朋友，也可以直接拿来使用，只需要扩展一些功能，即可写出比较不错的论文。
（5）或者从事开发的朋友，可以将该软件作为一个项目案例，也非常不错。

~~~


