<template>
    <div>
        <div class="page-header clear-filter">
            <parallax
                    class="page-header-image"
                    style="background:linear-gradient(#8e8e8e 0%, #ffffff 80%)">
            </parallax>
            <div class="container">
                <div class="content-center brand" v-if="response()">
                    <h1 class="h1-seo">AI 4 Covid</h1>
                    <h3>AI powered COVID-19 diagnosis</h3>
                    <input type="file" ref="file" style="display: none" @change="handleUpload()">
                    <n-button type="primary" @click="$refs.file.click()" size="lg">Upload X-ray</n-button>
                </div>

                <div v-else class="content-center brand" style="top: 50%">
                    <h1 class="h1-seo">Results</h1>
                    <card>
                        <p style="color: #2c2c2c; font-size: 1.5em" >
                            Risk of COVID-19: <b>{{data.prediction.toFixed(2)*100 + '%'}}</b>
                        </p>
                        <img :src="image.src" style="height:50%;width:50%"/>
                        <br>
                        <br>
                        <p style="color: #2c2c2c; font-size: 1em">
                            Decision heatmap highlights the key regions on the X-Ray <br> that contribute to the diagnosis.
                        </p>
                    </card>
                    <input type="file" ref="file" style="display: none" @change="handleUpload()">
                    <n-button type="primary" @click="$refs.file.click()" size="lg">Upload another</n-button>
                </div>
            </div>
        </div>
    </div>
</template>
<script>
    import {Parallax, Card} from '@/components';
    import Button from '../components/Button';
    import axios from 'axios';

    export default {
        name: 'index',
        bodyClass: 'index-page',
        components: {
            [Button.name]: Button,
            Parallax,
            Card,
        },
        methods: {
            handleUpload() {
                let formData = new FormData();
                let headers = {
                    'Content-Type': 'multipart/form-data',
                    'Access-Control-Allow-Origin': '*'
                };
                formData.append('file', this.$refs.file.files[0]);
                axios.post('http://localhost:8000/api/v1/classify/', formData, {headers}).then(res => {
                    // this.$router.push({name: 'results', params: {base64: res.data}});
                    this.data = res.data;
                    this.image = new Image();
                    this.image.src = 'data:image/jpg;base64,' + res.data.heatmap;
                }).catch(e => console.log(e));
            },
            response() {
                return this.data == '';
            }
        },
        data() {
            return {
                file: '',
                data: '',
                image: '',
            }
        }
    };
</script>
<style></style>
