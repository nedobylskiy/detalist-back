/**
 iZ³ | Izzzio blockchain - https://izzz.io

 Copyright 2018 Izio Ltd (OOO "Изио")

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

const APP_CONFIG = require('./appConfig.json');

/**
 * Detalist contract address
 * @type {string}
 */
const DETALIST_CONTRACT_ADDRESS = '1';

const logger = new (require(global.PATH.mainDir + '/modules/logger'))("Detalist");
const storj = require(global.PATH.mainDir + '/modules/instanceStorage');
const DApp = require(global.PATH.mainDir + '/app/DApp');
const TokenContractConnector = require(global.PATH.mainDir + '/modules/smartContracts/connectors/TokenContractConnector');
const fs = require('fs');


let that;

const express = require('express');
const app = express();
const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({
    extended: false
}));
app.use(bodyParser.json());

app.use('/', express.static('front/build'));
app.use('/front', express.static('detalist-front'));

const request = require('request');

/**
 * Detalist DApp
 * Provides API interface for DApp
 */
class App extends DApp {


    /**
     * Initialize
     */
    init() {
        that = this;
        app.listen(APP_CONFIG.httpPort, function () {
            logger.info('Detalist started at ' + APP_CONFIG.httpPort);
        });


        logger.info('Detalist DApp started');

        /*process.on('SIGINT', () => {
            console.log('Terminating tests...');
            process.exit();
        });

        process.on('unhandledRejection', error => {
            console.log(error);
            logger.fatal(error);
            process.exit();
        });*/


        app.get('/analytics/:id', async function (req, res) {
            let id = req.params.id;
            try {
                let result = await that.contracts.ecmaPromise.callMethodRollback(DETALIST_CONTRACT_ADDRESS, 'getItem', [id], {});
                request.post({url: 'http://localhost:5000/', form: {tree: result}}, function (err, httpResponse, body) {
                    //console.log(err, body);

                    res.send('data:image/png;base64,' + body.toString());
                })
            } catch (e) {
                res.send('Error');
            }
        });

        /**
         * Get item
         */
        app.get('/item/:id', async function (req, res) {
            let id = req.params.id;
            try {
                let result = await that.contracts.ecmaPromise.callMethodRollback(DETALIST_CONTRACT_ADDRESS, 'getItem', [id], {});
                res.send(that.jsonOkResponse(JSON.parse(result)))
            } catch (e) {
                res.send(that.jsonErrorResponse(e));
            }
        });

        /**
         * Get item
         */
        app.get('/item/bycode/:id', async function (req, res) {
            let id = req.params.id;
            try {
                let result = await that.contracts.ecmaPromise.callMethodRollback(DETALIST_CONTRACT_ADDRESS, 'getItemByCode', [id], {});
                res.send(that.jsonOkResponse(JSON.parse(result)))
            } catch (e) {
                res.send(that.jsonErrorResponse(e));
            }
        });

        app.get('/items', async function (req, res) {
            try {
                let result = await that.contracts.ecmaPromise.callMethodRollback(DETALIST_CONTRACT_ADDRESS, 'getAllItems', [], {});
                res.send(that.jsonOkResponse(JSON.parse(result)))
            } catch (e) {
                res.send(that.jsonErrorResponse(e));
            }
        });

        app.get('/items/search/:search', async function (req, res) {
            let search = req.params.search;
            that.ecmaContract.events.rawQuery('SELECT * FROM `events` WHERE `event` = ? AND contract = ? AND (`v4` like "%' + search + '%" OR `v5` like "%' + search + '%")', ['AddItem', DETALIST_CONTRACT_ADDRESS], function (err, data) {
                data = data.map(function (val) {
                    return {
                        id: val.v1,
                        code: val.v2,
                        bench: val.v3
                    };
                });

                res.send(that.jsonOkResponse(data));
            });
        });

        /**
         * New freelancer status
         */
        app.post('/item/create', async function (req, res) {
            console.log(req.body);
            let code = req.body.code;
            let type = req.body.type;
            let addedBy = req.body.addedBy;
            let bench = req.body.bench;
            let params = req.body.params; //TODO: Чблин делать?
            try {
                params = JSON.parse(params);
            } catch (e) {
            }
            let parts = [];

            if(Array.isArray(req.body['parts[]'])) {
                parts = req.body['parts[]'];
            } else if(req.body['parts[]']) {
                parts = [req.body['parts[]']];
            }

            let additionalInfo = req.body.additionalInfo;
            try {
                additionalInfo = JSON.parse(additionalInfo);
            } catch (e) {
            }


            if(!code || !type || !addedBy || !bench || !params || !Array.isArray(parts)) {
                return res.send(that.jsonErrorResponse('One of requied param not found'))
            }

            try {
                let result = await that.contracts.ecmaPromise.deployMethod(DETALIST_CONTRACT_ADDRESS, 'addItem', [code, type, addedBy, bench, params, parts, additionalInfo], {});
                res.send(that.jsonOkResponse({newBlock: result}))
            } catch (e) {
                res.send(that.jsonErrorResponse(e));
            }
        });


        app.post('/item/markBroken', async function (req, res) {
            let code = req.body.code;
            if(!code) {
                return res.send(that.jsonErrorResponse('Code param not found'))
            }
            try {
                let result = await that.contracts.ecmaPromise.deployMethod(DETALIST_CONTRACT_ADDRESS, 'markBrokenByCode', [code], {});
                res.send(that.jsonOkResponse({newBlock: result}))
            } catch (e) {
                res.send(that.jsonErrorResponse(e));
            }
        });

    }


    /**
     * Error response
     * @param error
     * @return {{errorMessage: *, error: boolean}}
     */
    jsonErrorResponse(error) {
        if(typeof error === 'string') {
            return {error: true, errorMessage: error};
        }
        return {error: true, errorMessage: error.message};
    }

    /**
     * Reponse ok
     * @param data
     * @return {{data: *, error: boolean}}
     */
    jsonOkResponse(data) {
        return {error: false, data: data}
    }


}

module.exports = App;
