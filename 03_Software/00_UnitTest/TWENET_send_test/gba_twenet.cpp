#include <TWELITE>
#include <NWK_SIMPLE>
#include <stdio.h>
#include "gba_twenet.h"

void gbaTwenetInit(uint8_t id, uint8_t channel, mwx::twenet* twelite){
	*twelite
		<< TWENET::appid(APP_ID)    
		<< TWENET::channel(channel) 
		<< TWENET::rx_when_idle();

	auto&& nwksmpl = twelite->network.use<NWK_SIMPLE>();
	nwksmpl << NWK_SIMPLE::logical_id(id) 
	        << NWK_SIMPLE::repeat_max(REPEAT_MAX);   

	twelite->begin(); 
}

bool gbaTwenetTransmit(GBA_DATA *gba_data, uint32_t addr, mwx::twenet* twelite){
	uint8_t dest[MES_LEN] = {};

	if (auto&& pkt = twelite->network.use<NWK_SIMPLE>().prepare_tx_packet()) {
		pkt << tx_addr(addr)
			<< tx_retry(RETRY_NUM)
			<< tx_packet_delay(TX_DELAY_ST,TX_DELAY_ED,TX_DELAY_INT);
		
			uint8_t *src = (uint8_t*)gba_data;

		for(size_t i=0; i<MES_LEN; i++){
			dest[i] = src[i];
		}

		pack_bytes(pkt.get_payload()
			, dest
		);

		pkt.transmit();
        return true;
	}
    else{
        return false;
    }
}

void gbaTwenetReceive(GBA_DATA* gba_data, RX_INFO* rx_info, mwx::packet_rx* pkt){

	uint8_t *src = pkt->get_payload().begin();
	uint8_t *dest = (uint8_t*)gba_data;

	for(size_t i=0; i<MES_LEN; i++){
		dest[i] = src[i];
	}	

    rx_info->receiveTime = pkt->get_psRxDataApp()->u32Tick;
    rx_info->cmd = pkt->get_psRxDataApp()->u8Cmd;
    rx_info->srcAddr = pkt->get_addr_src_lid();
    rx_info->seq = pkt->get_psRxDataApp()->u8Seq;
    rx_info->lqi = pkt->get_psRxDataApp()->u8Lqi;

}
